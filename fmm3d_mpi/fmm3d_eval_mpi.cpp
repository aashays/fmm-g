/* Kernel Independent Fast Multipole Method
   Copyright (C) 2004 Lexing Ying, New York University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */
#include "fmm3d_mpi.hpp"
#include "common/vecmatop.hpp"
#include "manage_petsc_events.hpp"
#include "p3d/point3d.h"
#include "p3d/upComp.h"
#include "p3d/dnComp.h"
#include "gpu_setup.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

using std::cerr;
using std::endl;

// ----------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::evaluate"
int FMM3d_MPI::evaluate(Vec srcDen, Vec trgVal)
{
#ifdef HAVE_PAPI
  // these variables are for use with PAPI
  float papi_real_time, papi_proc_time, papi_mflops;
  long_long papi_flpops=0, papi_flpops2;
  int papi_retval;
#endif

  PetscLogEventBegin(EvalIni_event,0,0,0,0);
  //begin  //ebiLogInfo( "multiply.............");
  //-----------------------------------
  //cerr<<"fmm src and trg numbers "<<pglbnum(_srcPos)<<" "<<pglbnum(_trgPos)<<endl;
  PetscInt tmp;
  pC( VecGetSize(srcDen,&tmp) );  pA(tmp==srcDOF()*procGlbNum(_srcPos));
  pC( VecGetSize(trgVal,&tmp) );  pA(tmp==trgDOF()*procGlbNum(_trgPos));

  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();

  // shall we skip all communication? (results will be incorrect, of course)
  PetscTruth skip_communication;
  PetscOptionsHasName(0,"-eval_skip_communication",&skip_communication);
  if (skip_communication && !mpiRank())
    std::cout<<"!!!!! All communications during interaction evaluation are skipped. Results are incorrect !!!!"<<endl;


  //1. zero out vecs.  This includes all global, contributor, user, evaluator vectors.
  PetscScalar zero=0.0;
  pC( VecSet(trgVal, zero) );
  pC( VecSet(_glbSrcExaDen, zero) );
  pC( VecSet(_glbSrcUpwEquDen, zero) );
  pC( VecSet(_ctbSrcExaDen, zero) );
  pC( VecSet(_ctbSrcUpwEquDen, zero) );
  pC( VecSet(_ctbSrcUpwChkVal, zero) );
  pC( VecSet(_usrSrcExaDen, zero) );
  pC( VecSet(_usrSrcUpwEquDen, zero) );
  pC( VecSet(_evaTrgExaVal, zero) );
  pC( VecSet(_evaTrgDwnEquDen, zero) );
  pC( VecSet(_evaTrgDwnChkVal, zero) );

  vector<int> ordVec;
  pC( _let->upwOrderCollect(ordVec) ); //BOTTOM UP collection of nodes

  //2. for contributors, load exact densities
  PetscInt procLclStart, procLclEnd; _let->procLclRan(_srcPos, procLclStart, procLclEnd);
  double* darr; pC( VecGetArray(srcDen, &darr) );
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if(_let->node(gNodeIdx).tag() & LET_CBTRNODE) {
		if(_let->terminal(gNodeIdx)==true) {
		  DblNumVec ctbSrcExaDen(this->ctbSrcExaDen(gNodeIdx));
		  vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).ctbSrcOwnVecIdxs();
		  for(size_t k=0; k<curVecIdxs.size(); k++) {
			 PetscInt poff = curVecIdxs[k] - procLclStart;
			 for(int d=0; d<srcDOF; d++) {
				ctbSrcExaDen(k*srcDOF+d) = darr[poff*srcDOF+d];
			 }
		  }
		}
	 }
  }
  pC( VecRestoreArray(srcDen, &darr) );
  PetscLogEventEnd(EvalIni_event,0,0,0,0);

  if (!skip_communication)
  {
    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin
    PetscLogEventBegin(EvalCtb2GlbExa_event,0,0,0,0);
    // send source densities from contributors to owners; this now should not involve any MPI communication, since for all leaf nodes in global tree owners are the only contributors; maybe eventually I'll remove this scatter at all
    pC( VecScatterBegin(_ctb2GlbSrcExaDen, _ctbSrcExaDen, _glbSrcExaDen,    ADD_VALUES, SCATTER_FORWARD) );
    pC( VecScatterEnd(_ctb2GlbSrcExaDen,  _ctbSrcExaDen, _glbSrcExaDen,    ADD_VALUES, SCATTER_FORWARD) );
    PetscLogEventEnd(EvalCtb2GlbExa_event,0,0,0,0);

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin
    PetscLogEventBegin(EvalGlb2UsrExaBeg_event,0,0,0,0);
    // we overlap sending of charge densities from owners to users with upward computation
    pC( VecScatterBegin(_usr2GlbSrcExaDen, _glbSrcExaDen, _usrSrcExaDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrExaBeg_event,0,0,0,0);
  }

  //3. up computation
  PetscLogEventBegin(EvalUpwComp_event,0,0,0,0);
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  upComp_t *UpC;
  PetscTruth gpu_s2m;
  PetscOptionsHasName(0,"-gpu_s2m",&gpu_s2m);
  if (gpu_s2m)
    // compute s2m for all leaves at once
  {
    /* Allocate memory for the upward computation structure for GPU */
    if ( (UpC = (upComp_t*) calloc (1, sizeof (upComp_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for upward computation structure\n");
      return 1;
    }		//why??
    /* Copy data into the upward computation structure defined by 'UpC' */
    UpC->tag = UC;
    UpC->numSrc = procLclNum(_usrSrcExaPos);
    UpC->dim = 3;
    UpC->numSrcBox = ordVec.size();
    // samPos = this->matmgnt()->samPos(UpC->tag);
    const DblNumMat & sample_pos = _matmgnt->samPos(UpC->tag);
    vector<float> sample_pos_float(sample_pos.n()*sample_pos.m());
    for (size_t i=0; i<sample_pos_float.size(); i++)
      sample_pos_float[i]=*(sample_pos._data+i);

    UpC->src_ = (float *) malloc(sizeof(float) * UpC->numSrc * (UpC->dim+1));
    UpC->trgVal = (float**) malloc (sizeof(float*) * ordVec.size());
    UpC->srcBoxSize = (int *) calloc (ordVec.size(), sizeof(int));
    UpC->trgCtr = (float *) calloc (UpC->numSrcBox * UpC->dim, sizeof(float));
    UpC->trgRad = (float *) calloc (UpC->numSrcBox, sizeof(float));
    UpC->trgDim=sample_pos.n();
    UpC->samPosF=&sample_pos_float[0];

    int srcIndex = 0;
    for (size_t gNodeIdx=0; gNodeIdx<_let->nodeVec().size(); gNodeIdx++)
    {
      UpC->trgVal[gNodeIdx] = NULL;
      if (_let->terminal(gNodeIdx)   &&   _let->node(gNodeIdx).tag() & LET_CBTRNODE)
      {
	for (int j = 0; j < UpC->dim; j++)
	  UpC->trgCtr[j+gNodeIdx*UpC->dim] = _let->center(gNodeIdx)(j);

	/* Radius of the box */
	UpC->trgRad[gNodeIdx] = _let->radius(gNodeIdx);

	/* Allocate memory for target potentials */
	UpC->trgVal[gNodeIdx] = (float *) calloc(UpC->trgDim, sizeof(float));

	/* Source points and density stored as x1 y1 z1 d1 x2 y2 z2 d2 ..... */
	DblNumMat sources = ctbSrcExaPos(gNodeIdx);
	DblNumVec densities = ctbSrcExaDen(gNodeIdx);
	UpC->srcBoxSize[gNodeIdx] = sources.n();
	for(int s = 0; s < UpC->srcBoxSize[gNodeIdx]; s++) {
	  for(int d = 0; d < UpC->dim; d++)
	    UpC->src_[(s*(UpC->dim+1))+d+srcIndex] = sources(d,s);
	  UpC->src_[(s*(UpC->dim+1))+3+srcIndex] = densities(s);
	}
	srcIndex += (UpC->srcBoxSize[gNodeIdx] * (UpC->dim+1));
      }
    }

    gpu_up(UpC);
  }

  for(size_t i=0; i<ordVec.size(); i++) {
    int gNodeIdx = ordVec[i];
    if( _let->node(gNodeIdx).tag() & LET_CBTRNODE) {
      if(_let->depth(gNodeIdx)>=0) {
	DblNumVec ctbSrcUpwChkValgNodeIdx(ctbSrcUpwChkVal(gNodeIdx));
	DblNumVec ctbSrcUpwEquDengNodeIdx(ctbSrcUpwEquDen(gNodeIdx));
	if(_let->terminal(gNodeIdx)==true)
	{
	  if (gpu_s2m)
	  {
	    for (int j = 0; j < ctbSrcUpwChkValgNodeIdx.m(); j++)
	      ctbSrcUpwChkValgNodeIdx(j) = UpC->trgVal[gNodeIdx][j];
	  }
	  else
	  {
	    //S2M
	    pC( SrcEqu2UpwChk_dgemv(ctbSrcExaPos(gNodeIdx), ctbSrcExaNor(gNodeIdx), _let->center(gNodeIdx), _let->radius(gNodeIdx), ctbSrcExaDen(gNodeIdx), ctbSrcUpwChkValgNodeIdx) );
	  }
	}
	else
	{
	  //M2M
	  for(int a=0; a<2; a++) for(int b=0; b<2; b++) for(int c=0; c<2; c++) {
	    Index3 idx(a,b,c);
	    int chi = _let->child(gNodeIdx, idx);
	    if(_let->node(chi).tag() & LET_CBTRNODE) {
	      pC( _matmgnt->UpwEqu2UpwChk_dgemv(_let->depth(chi)+_rootLevel, idx, ctbSrcUpwEquDen(chi), ctbSrcUpwChkValgNodeIdx) );
	    }
	  }
	}
	//M2M
	pC( _matmgnt->UpwChk2UpwEqu_dgemv(_let->depth(gNodeIdx)+_rootLevel, ctbSrcUpwChkValgNodeIdx, ctbSrcUpwEquDengNodeIdx) );
      }
    }
  }

  if(gpu_s2m)
  {
    free (UpC->src_);
    free (UpC->srcBoxSize);
    free (UpC->trgCtr);
    free (UpC->trgRad);
    for (int i = 0; i < ordVec.size(); i++)
      free (UpC->trgVal[ordVec[i]]);
    free (UpC->trgVal);
    free (UpC);
  }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  PetscLogEventEnd(EvalUpwComp_event,0,0,0,0);

  //4. vectbscatters
  if (!skip_communication)
  {
    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalCtb2GlbEqu_event,0,0,0,0);
    pC( VecScatterBegin( _ctb2GlbSrcUpwEquDen, _ctbSrcUpwEquDen, _glbSrcUpwEquDen,    ADD_VALUES, SCATTER_FORWARD) );
    pC( VecScatterEnd(_ctb2GlbSrcUpwEquDen,   _ctbSrcUpwEquDen, _glbSrcUpwEquDen,    ADD_VALUES, SCATTER_FORWARD) );
    PetscLogEventEnd(EvalCtb2GlbEqu_event,0,0,0,0);

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalGlb2UsrEquBeg_event,0,0,0,0);
    // sending equiv. densities from owners to users is overlapped with U-list computations
    pC( VecScatterBegin(_usr2GlbSrcUpwEquDen, _glbSrcUpwEquDen, _usrSrcUpwEquDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrEquBeg_event,0,0,0,0);

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalGlb2UsrExaEnd_event,0,0,0,0);
    // we overlap sending of charge densities from owners to users with upward computation (scatterBegin is several lines above)
    pC( VecScatterEnd(_usr2GlbSrcExaDen, _glbSrcExaDen, _usrSrcExaDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrExaEnd_event,0,0,0,0);
  }

  // U-list computation
  PetscLogEventBegin(EvalUList_event,0,0,0,0);
  PetscTruth gpu_ulist;
  PetscOptionsHasName(0,"-gpu_ulist",&gpu_ulist);
  if (gpu_ulist)
  {
    // Interface U-list contribution calculation for GPU
    point3d_t *P;
    if ( (P = (point3d_t*) malloc (sizeof (point3d_t))) == NULL ) {
      fprintf (stderr, " Error allocating memory for u-list structure\n");
      return 1;
    }
    // Copy data into the u-list structure defined by 'P'

    // P->numSrc = (*_srcPos).n();
    P->numSrc =procLclNum(_usrSrcExaPos) ;
    P->numTrg = procLclNum(_evaTrgExaPos);
    P->dim = 3;

    P->src_ = (float *) malloc(sizeof(float) * P->numSrc * (P->dim+1));
    P->trg_ = (float *) malloc(sizeof(float) * P->numTrg * P->dim);
    P->trgVal = (float *) calloc(P->numTrg, sizeof(float));

    P->uList = (int **) malloc (sizeof(int*) * ordVec.size());
    P->uListLen = (int *) calloc (ordVec.size(), sizeof(int));
    P->srcBoxSize = (int *) calloc (ordVec.size(), sizeof(int));
    P->trgBoxSize = (int *) calloc (ordVec.size(), sizeof(int));

    P->numTrgBox = ordVec.size();
    P->numSrcBox = ordVec.size();		// TODO: Are the total number of source and target boxes always the same?
    int j;
    int trgIndex = 0;
    int srcIndex = 0;
    int tv = 0;
    int d = 0;
    for(int i=ordVec.size()-1; i >= 0; i--) {
      int gNodeIdx = ordVec[i];
      P->uList[gNodeIdx] = NULL;
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  Let3d_MPI::Node& curNode = _let->node(gNodeIdx);
	  P->uList[gNodeIdx] = (int*) malloc (sizeof(int) * curNode.Unodes().size());
	  P->uListLen[gNodeIdx] = curNode.Unodes().size();
	  j = 0;
	  for(vector<int>::iterator vi=curNode.Unodes().begin(); vi!=curNode.Unodes().end(); vi++) {
	    P->uList[gNodeIdx][j] = *vi;
	    j++;
	  }
	  // P->trgBoxSize[gNodeIdx] = curNode.evaTrgExaNum();
	  DblNumMat evaTrgExaPosgNodeIdx(evaTrgExaPos(gNodeIdx));
	  P->trgBoxSize[gNodeIdx] =  evaTrgExaPosgNodeIdx.n();  // curNode.evaTrgExaNum();
	  assert (evaTrgExaPosgNodeIdx.n() == curNode.evaTrgExaNum());
	  for(int t = 0; t < P->trgBoxSize[gNodeIdx]; t++) {
	    for(d = 0; d < P->dim; d++)
	    {
	      // std::cout<<evaTrgExaPosgNodeIdx(d,t)<<" ";
	      P->trg_[(t*P->dim)+d+trgIndex] =  evaTrgExaPosgNodeIdx(d,t);
	    }
	    // std::cout<<endl;
	  }
	}
	trgIndex += (P->trgBoxSize[gNodeIdx] * P->dim);
	tv += P->trgBoxSize[gNodeIdx];
      }

      if( _let->node(gNodeIdx).tag() & LET_USERNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  P->srcBoxSize[gNodeIdx] = _let->node(gNodeIdx).usrSrcExaNum();
	  for(int s = 0; s < P->srcBoxSize[gNodeIdx]; s++) {
	    for(d = 0; d < P->dim; d++)
	      P->src_[(s*(P->dim+1))+d+srcIndex] = (usrSrcExaPos(gNodeIdx)(d,s));
	    P->src_[(s*(P->dim+1))+d+srcIndex] = usrSrcExaDen(gNodeIdx)(s);
	  }
	}
	srcIndex += (P->srcBoxSize[gNodeIdx] * (P->dim+1));
      }
    }

    //  Calculate dense interations
    dense_inter_gpu(P);

    trgIndex = 0;
    // Copy target potentials back into the original structure
    // * for use by rest of the algorithm
    for(int i=ordVec.size()-1; i >= 0; i--) // actually any order is fine
    {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  for(int t = 0; t < P->trgBoxSize[gNodeIdx]; t++) {
	     // std::cout<<P->trgVal[t+trgIndex]<<" ";
	    evaTrgExaVal(gNodeIdx)(t)= P->trgVal[t+trgIndex];
	  }
	}
      }
      trgIndex += P->trgBoxSize[gNodeIdx];
    }

    // Free memory allocated for the interface
    free (P->src_);
    free (P->trg_);
    free (P->trgVal);
    free (P->uListLen);
    free (P->srcBoxSize);
    free (P->trgBoxSize);
    for(int i=ordVec.size()-1; i >= 0; i--)
      free (P->uList[ordVec[i]]);
    free (P->uList);
    free (P);
  }
  else
  {
#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
    for(size_t i=0; i<ordVec.size(); i++) {
      int gNodeIdx = ordVec[i];
      if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
	if( _let->terminal(gNodeIdx)==true ) { //terminal
	  DblNumVec evaTrgExaValgNodeIdx(evaTrgExaVal(gNodeIdx));
	  DblNumMat evaTrgExaPosgNodeIdx(evaTrgExaPos(gNodeIdx));
	  for(vector<int>::iterator vi=_let->node(gNodeIdx).Unodes().begin(); vi!=_let->node(gNodeIdx).Unodes().end(); vi++) {
	    //S2T
	    pC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPosgNodeIdx, usrSrcExaDen(*vi), evaTrgExaValgNodeIdx) );
	  }
	}
      }
    }
#ifdef HAVE_PAPI
    // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
    // papi_real_time, papi_proc_time, papi_mflops are just discarded
    if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
      SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
    PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  }
  PetscLogEventEnd(EvalUList_event,0,0,0,0);

  if (!skip_communication)
  {
    PetscLogEventBegin(EvalGlb2UsrEquEnd_event,0,0,0,0);
    // sending equiv. densities from owners to users is overlapped with U-list computations (scatterBegin is several lines above)
    pC( VecScatterEnd(_usr2GlbSrcUpwEquDen, _glbSrcUpwEquDen, _usrSrcUpwEquDen, INSERT_VALUES, SCATTER_REVERSE) );
    PetscLogEventEnd(EvalGlb2UsrEquEnd_event,0,0,0,0);
  }

  //V
  PetscLogEventBegin(EvalVList_event,0,0,0,0);
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  for(size_t i=0; i<ordVec.size(); i++) {
    int gNodeIdx = ordVec[i];
    if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
      Point3 gNodeIdxctr(_let->center(gNodeIdx));
      double D = 2.0 * _let->radius(gNodeIdx);

      DblNumVec evaTrgDwnChkVal(this->evaTrgDwnChkVal(gNodeIdx));
      for(vector<int>::iterator vi=_let->node(gNodeIdx).Vnodes().begin(); vi!=_let->node(gNodeIdx).Vnodes().end(); vi++) {
	Point3 victr(_let->center(*vi));
	Index3 idx;		  for(int d=0; d<dim(); d++)			 idx(d) = int(floor( (victr[d]-gNodeIdxctr[d])/D+0.5));

	Node& srcnode = node(*vi);
	Node& trgnode = node(gNodeIdx);
	if(srcnode.vLstOthCnt()==0) {
	  srcnode.effDen().resize( _matmgnt->effDatSze(UE) );			 setvalue(srcnode.effDen(), 0.0);   //1. resize effDen
	  pC( _matmgnt->plnDen2EffDen(_let->depth(gNodeIdx)+_rootLevel, usrSrcUpwEquDen(*vi),  srcnode.effDen()) ); //2. transform from UpwEquDen to effDen
	}
	if(trgnode.vLstInCnt()==0) {
	  trgnode.effVal().resize( _matmgnt->effDatSze(DC) );			 setvalue(trgnode.effVal(), 0.0); //1. resize effVal
	}
	//M2L
	pC( _matmgnt->UpwEqu2DwnChk_dgemv(_let->depth(gNodeIdx)+_rootLevel, idx, srcnode.effDen(), trgnode.effVal()) );

	srcnode.vLstOthCnt()++;
	trgnode.vLstInCnt()++;
	if(srcnode.vLstOthCnt()==srcnode.vLstOthNum()) {
	  srcnode.effDen().resize(0); //1. resize effDen to 0
	  srcnode.vLstOthCnt()=0;
	}
	if(trgnode.vLstInCnt()==trgnode.vLstInNum()) {
	  pC( _matmgnt->effVal2PlnVal(_let->depth(gNodeIdx)+_rootLevel, trgnode.effVal(), evaTrgDwnChkVal) ); //1. transform from effval to DwnChkVal
	  trgnode.effVal().resize(0); //2. resize effVal to 0
	  trgnode.vLstInCnt()=0;
	}
      }
    }
  }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  PetscLogEventEnd(EvalVList_event,0,0,0,0);

  //W
  PetscLogEventBegin(EvalWList_event,0,0,0,0);
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  for(size_t i=0; i<ordVec.size(); i++) {
    int gNodeIdx = ordVec[i];
    if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
      if( _let->terminal(gNodeIdx)==true ) {
	DblNumVec evaTrgExaVal_gNodeIdx(this->evaTrgExaVal(gNodeIdx));
	for(vector<int>::iterator vi=_let->node(gNodeIdx).Wnodes().begin(); vi!=_let->node(gNodeIdx).Wnodes().end(); vi++) {

	  // terminal nodes in LET might be parent nodes in global tree;
	  // thus, in some cases we instead need to check glbSrcExaNum or glbSrcExaBeg;
	  // both are guaranteed to be -1 for parent nodes in global tree
	  // and both guaranteed to be >=0 for leaves in global tree
	  if(_let->node(*vi).glbSrcExaBeg()>=0 && _let->node(*vi).usrSrcExaNum()*srcDOF<_matmgnt->plnDatSze(UE)) { //use Exa instead
	    //S2T
	    pC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPos(gNodeIdx), usrSrcExaDen(*vi), evaTrgExaVal_gNodeIdx) );
	  } else {
	    //M2T
	    int vni = *vi;
	    pC( UpwEqu2TrgChk_dgemv(_let->center(vni), _let->radius(vni), evaTrgExaPos(gNodeIdx), usrSrcUpwEquDen(*vi), evaTrgExaVal_gNodeIdx) );
	  }
	}
      }
    }
  }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  PetscLogEventEnd(EvalWList_event,0,0,0,0);

  //X
  PetscLogEventBegin(EvalXList_event,0,0,0,0);
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
		DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
		DblNumVec evaTrgDwnChkVal_gNodeIdx(evaTrgDwnChkVal(gNodeIdx));
		for(vector<int>::iterator vi=_let->node(gNodeIdx).Xnodes().begin(); vi!=_let->node(gNodeIdx).Xnodes().end(); vi++) {
		  if(_let->terminal(gNodeIdx) && _let->node(gNodeIdx).evaTrgExaNum()*trgDOF<_matmgnt->plnDatSze(DC)) { //use Exa instead
			 pC( SrcEqu2TrgChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), evaTrgExaPos(gNodeIdx), usrSrcExaDen(*vi), evaTrgExaVal_gNodeIdx) );
		  } else {
			 //S2L
			 pC( SrcEqu2DwnChk_dgemv(usrSrcExaPos(*vi), usrSrcExaNor(*vi), _let->center(gNodeIdx), _let->radius(gNodeIdx), usrSrcExaDen(*vi), evaTrgDwnChkVal_gNodeIdx) );
		  }
		}
	 }
  }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  PetscLogEventEnd(EvalXList_event,0,0,0,0);

  //7. combine
  PetscLogEventBegin(EvalCombine_event,0,0,0,0);
  ordVec.clear();  pC( _let->dwnOrderCollect(ordVec) );
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
#endif
  dnComp_t *DnC;
  PetscTruth gpu_l2t;
  PetscOptionsHasName(0,"-gpu_l2t",&gpu_l2t);
  if (gpu_l2t) {

//	  dnComp_t *DnC;
	   if ( (DnC = (dnComp_t*) calloc (1, sizeof (dnComp_t))) == NULL ) {
	 	fprintf (stderr, " Error allocating memory for downward computation structure\n");
	         return 1;
	   }
	   /* Copy data into the downward computation structure defined by 'DnC' */
	   DnC->tag = DE;
	   DnC->numTrg = procLclNum(_evaTrgExaPos);
	   DnC->dim = 3;
	   DnC->numTrgBox = ordVec.size();		//ordvec correct

	   DnC->trg_ = (float *) malloc(sizeof(float) * DnC->numTrg * DnC->dim);
	   DnC->trgVal = (float**) malloc (sizeof(float*) * DnC->numTrgBox );
	   DnC->trgBoxSize = (int *) calloc (ordVec.size(), sizeof(int));
	   DnC->srcCtr = (float *) calloc (DnC->numTrgBox * DnC->dim, sizeof(float));
	   DnC->srcRad = (float *) calloc (DnC->numTrgBox, sizeof(float));
//	   samPos = this->matmgnt()->samPos(DnC->tag);

		const DblNumMat & sample_pos = _matmgnt->samPos(DnC->tag);		//copied from s2m
		vector<float> sample_pos_float(sample_pos.n()*sample_pos.m());
		for (size_t i=0; i<sample_pos_float.size(); i++)
			sample_pos_float[i]=*(sample_pos._data+i);

//	   DnC->srcDim=samPos.n()*srcDOF;
//	   DnC->samPosF=samPos._data;
	   DnC->srcDim=sample_pos.n();
	   DnC->samPosF=&sample_pos_float[0];
	   DnC->srcDen=(float*)calloc(DnC->numTrgBox*DnC->srcDim,sizeof(float));

//	   int dd=0;
//	   int trgIndex = 0;
//	   for(int i=0; i<ordVec.size(); i++) {
//	 	 int gNodeIdx = ordVec[i];
//	      DnC->trgVal[gNodeIdx] = NULL;
//	 	 if(  _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //eValuator
//	 		if(_let->terminal(gNodeIdx)) {
//	 		  //L2T - local -> target
//	 		  DblNumVec trgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
////	 #ifdef DS_ORG
//	 			/* Center of the box */
//	 			for (int j = 0; j < DnC->dim; j++)
//	 				DnC->srcCtr[j+gNodeIdx*DnC->dim] = _let->center(gNodeIdx)(j);
//
//	 			/* Radius of the box */
//	 			DnC->srcRad[gNodeIdx] = _let->radius(gNodeIdx);
//
//	 			/* Allocate memory for target potentials */
//	   			DnC->trgVal[gNodeIdx] = (float *) calloc(trgExaVal_gNodeIdx.m(), sizeof(float));
//
//
//	   			DblNumMat targets = evaTrgExaPos(gNodeIdx);
//	   			DnC->trgBoxSize[gNodeIdx] = targets.n();
//	   			for(int s = 0; s < DnC->trgBoxSize[gNodeIdx]; s++) {
//	   			  for(int d = 0; d < DnC->dim; d++) {
//	   			    DnC->trg_[(s*(DnC->dim))+d+trgIndex] = targets(d,s);
////	   			  std::cout<<"trg: "<< DnC->trg_[(s*(DnC->dim))+d+trgIndex]<<endl;
//	   			  }
//	   			}
//
////	   			/* Target points are stored as x1 y1 z1 x2 y2 z2 ..... */
////	 			vector<int>& curVecIdxs = _let->node(gNodeIdx).trgOwnVecIdxs();
////	 	  		DnC->trgBoxSize[gNodeIdx] = curVecIdxs.size ();
////	           		for(int t = 0; t < DnC->trgBoxSize[gNodeIdx]; t++) {
////	 	     			for(d = 0; d < DnC->dim; d++){
////	 	     				DnC->trg_[(t*DnC->dim)+d+trgIndex] = (trgExaPos(gNodeIdx)(d,t));
////	 	     			}
////	           		}
//
//
//	           		for(int t = 0; t < DnC->srcDim; t++) {
//	           			DnC->srcDen[gNodeIdx*DnC->srcDim+t]=evaTrgDwnEquDen(gNodeIdx)(t);
////	           			std::cout<<"den "<<DnC->srcDen[gNodeIdx*DnC->srcDim+t]<<endl;
//	           		}
//
//	          		trgIndex += (DnC->trgBoxSize[gNodeIdx] * DnC->dim);
//	 //		  iC( DwnEqu2TrgChk_sgemv(_let->center(gNodeIdx), _let->radius(gNodeIdx), trgExaPos(gNodeIdx), trgDwnEquDen(gNodeIdx), trgExaVal_gNodeIdx) );
//	 		}
//	 	 }
//	   }

	 //////////Downward computation GPU call///////////
//	 #ifdef DS_ORG
//	 gpu_down(DnC);
  }	//end if l2t if
  int trgIndex = 0;
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
		if(_let->depth(gNodeIdx)>=3) {
		  int pargNodeIdx = _let->parent(gNodeIdx);
		  Index3 chdidx( _let->path2Node(gNodeIdx)-2 * _let->path2Node(pargNodeIdx) );
		  //L2L
		  DblNumVec evaTrgDwnChkVal_gNodeIdx(evaTrgDwnChkVal(gNodeIdx));
		  pC( _matmgnt->DwnEqu2DwnChk_dgemv(_let->depth(pargNodeIdx)+_rootLevel, chdidx, evaTrgDwnEquDen(pargNodeIdx), evaTrgDwnChkVal_gNodeIdx) );
		}
		if(_let->depth(gNodeIdx)>=2) {
		  //L2L
		  DblNumVec evaTrgDwnEquDen_gNodeIdx(evaTrgDwnEquDen(gNodeIdx));
		  pC( _matmgnt->DwnChk2DwnEqu_dgemv(_let->depth(gNodeIdx)+_rootLevel, evaTrgDwnChkVal(gNodeIdx), evaTrgDwnEquDen_gNodeIdx) );
		}
		if(_let->terminal(gNodeIdx)) {
		  //L2T
//		  DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
		  if (gpu_l2t)
		  				  {
//		  				    for (int j = 0; j < evaTrgExaVal_gNodeIdx.m(); j++) {
////		  				    	std::cout<<DnC->trgVal[gNodeIdx][j]<<" ";
//		  				    	evaTrgExaVal_gNodeIdx(j) += DnC->trgVal[gNodeIdx][j];
//		  				    }
	 		  //L2T - local -> target
	 		  DblNumVec trgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
//	 #ifdef DS_ORG
	 			/* Center of the box */
	 			for (int j = 0; j < DnC->dim; j++)
	 				DnC->srcCtr[j+gNodeIdx*DnC->dim] = _let->center(gNodeIdx)(j);

	 			/* Radius of the box */
	 			DnC->srcRad[gNodeIdx] = _let->radius(gNodeIdx);

	 			/* Allocate memory for target potentials */
	   			DnC->trgVal[gNodeIdx] = (float *) calloc(trgExaVal_gNodeIdx.m(), sizeof(float));


	   			DblNumMat targets = evaTrgExaPos(gNodeIdx);
	   			DnC->trgBoxSize[gNodeIdx] = targets.n();
	   			for(int s = 0; s < DnC->trgBoxSize[gNodeIdx]; s++) {
	   			  for(int d = 0; d < DnC->dim; d++) {
	   			    DnC->trg_[(s*(DnC->dim))+d+trgIndex] = targets(d,s);
//	   			  std::cout<<"trg: "<< DnC->trg_[(s*(DnC->dim))+d+trgIndex]<<endl;
	   			  }
	   			}

//	   			/* Target points are stored as x1 y1 z1 x2 y2 z2 ..... */
//	 			vector<int>& curVecIdxs = _let->node(gNodeIdx).trgOwnVecIdxs();
//	 	  		DnC->trgBoxSize[gNodeIdx] = curVecIdxs.size ();
//	           		for(int t = 0; t < DnC->trgBoxSize[gNodeIdx]; t++) {
//	 	     			for(d = 0; d < DnC->dim; d++){
//	 	     				DnC->trg_[(t*DnC->dim)+d+trgIndex] = (trgExaPos(gNodeIdx)(d,t));
//	 	     			}
//	           		}


	           		for(int t = 0; t < DnC->srcDim; t++) {
	           			DnC->srcDen[gNodeIdx*DnC->srcDim+t]=evaTrgDwnEquDen(gNodeIdx)(t);
//	           			std::cout<<"den "<<DnC->srcDen[gNodeIdx*DnC->srcDim+t]<<endl;
	           		}

	          		trgIndex += (DnC->trgBoxSize[gNodeIdx] * DnC->dim);
		  				  }
		  else {
			  DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
		  pC( DwnEqu2TrgChk_dgemv(_let->center(gNodeIdx), _let->radius(gNodeIdx), evaTrgExaPos(gNodeIdx), evaTrgDwnEquDen(gNodeIdx), evaTrgExaVal_gNodeIdx) );
			}
		}
	 }
  }
  if(gpu_l2t) {
	  gpu_down(DnC);

  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
		 if(_let->terminal(gNodeIdx)) {
			 DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
			 for (int j = 0; j < evaTrgExaVal_gNodeIdx.m(); j++) {
//				std::cout<<DnC->trgVal[gNodeIdx][j]<<" ";
				evaTrgExaVal_gNodeIdx(j) += DnC->trgVal[gNodeIdx][j];
			}

	 }
	 }
  }
  }
//  for(size_t i=0; i<ordVec.size(); i++) {
//	 int gNodeIdx = ordVec[i];
//	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) { //evaluator
//		 if(_let->terminal(gNodeIdx)) {
//			 DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
//			 for (int j = 0; j < evaTrgExaVal_gNodeIdx.m(); j++) {
//				std::cout<<evaTrgExaVal_gNodeIdx(j)<<" ";
////				evaTrgExaVal_gNodeIdx(j) += DnC->trgVal[gNodeIdx][j];
//			}
//			 free(DnC->trgVal[gNodeIdx]);
//	 }
//	 }
//	 std::cout<<endl;
//  }
  if(gpu_l2t) {
	  free (DnC->trg_);
	  free (DnC->trgBoxSize);
	  free (DnC->srcCtr);
	  free (DnC->srcRad);
//	  for (int i = 0; i < ordVec.size(); i++)
//		free (DnC->trgVal[ordVec[i]]);
	  free (DnC->trgVal);
	  free(DnC->srcDen);
	  free (DnC);
  }
#ifdef HAVE_PAPI
  // read flop counter from papi (first such call initializes library and starts counters; on first call output variables are apparently unchanged)
  // papi_real_time, papi_proc_time, papi_mflops are just discarded
  if ((papi_retval = PAPI_flops(&papi_real_time, &papi_proc_time, &papi_flpops2, &papi_mflops)) != PAPI_OK)
    SETERRQ1(1,"PAPI failed with errorcode %d",papi_retval);
  PetscLogFlops(papi_flpops2-papi_flpops);
#endif
  PetscLogEventEnd(EvalCombine_event,0,0,0,0);

  PetscLogEventBegin(EvalFinalize_event,0,0,0,0);
  //8. save tdtExaVal
  _let->procLclRan(_trgPos, procLclStart, procLclEnd);
  double* varr; pC( VecGetArray(trgVal, &varr) );
  for(size_t i=0; i<ordVec.size(); i++) {
	 int gNodeIdx = ordVec[i];
	 if( _let->node(gNodeIdx).tag() & LET_EVTRNODE ) {
		if( _let->terminal(gNodeIdx)==true ) {
		  DblNumVec evaTrgExaVal(this->evaTrgExaVal(gNodeIdx));
		  vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).evaTrgOwnVecIdxs();
		  for(size_t k=0; k<curVecIdxs.size(); k++) {
			 PetscInt poff = curVecIdxs[k] - procLclStart;
			 for(int d=0; d<trgDOF; d++) {
				varr[poff*trgDOF+d] = evaTrgExaVal(k*trgDOF+d);
			 }
		  }
		}
	 }
  }
  pC( VecRestoreArray(trgVal, &varr) );
  PetscLogEventEnd(EvalFinalize_event,0,0,0,0);

  // I don't understand the role of barrier below. Let's remove it and see if things break.
  // pC( MPI_Barrier(mpiComm()) );  //check vLstInCnt, vLstOthCnt
  return(0);
}



