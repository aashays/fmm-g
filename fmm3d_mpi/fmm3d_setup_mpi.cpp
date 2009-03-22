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
#include <time.h>
#include "manage_petsc_events.hpp"

using std::cerr;
using std::endl;

/** \file
 * fmm_setup_mpi.cpp sets up all of the evaluator, contributor and user
 * data, all essential for running the fmm code in parallel.  Further
 * comments for each section outlinhed below.
 */


// ----------------------------------------------------------------------
/* FMM setup sets up all of the essential information needed for
 * running the fast multipole 3d mpi implementation.  A major
 * componenet is the setup of the local essential tree, which
 * builds the octree, source data requirements and target data
 * requirements
 */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::setup"
int FMM3d_MPI::setup()
{
  //-----------------------------------------------------
  PetscTruth flg = PETSC_FALSE;
  /* Get the np variable.  Indicates level of proecision */
  // cerr<<"Prefix: "<<prefix()<<endl;
  PetscInt np_temp;
  pC( PetscOptionsGetInt(prefix().c_str(), "-np",     &np_temp,     &flg) );  
  pA(flg==true);
  _np = int(np_temp);
  
  //-----------------------------------------------------
  pA(_srcPos!=NULL && _srcNor!=NULL && _trgPos!=NULL);
  /* Build a new Communication Object for the local essential tree (LET)
	* and run setup on it.  See let3d_mpi.hpp and let3d_mpi.cpp for
	* more information */
  _let = new Let3d_MPI(prefix()+"let3d_");
  _let->srcPos()=_srcPos;
  _let->trgPos()=_trgPos;
  _let->ctr()= _ctr;  // Point3(0.5,0.5,0.5);
  // test:
  // _ctr = 0.55;
  _let->rootLevel()=_rootLevel; 

  PetscLogEventBegin(let_setup_event,0,0,0,0);
  pC( _let->setup() ); 
  PetscLogEventEnd(let_setup_event,0,0,0,0);

  // targets and sources were redistributed (new vectors were created and old ones were destroyed), so we should update our pointers
  _srcPos = _let->srcPos();
  _trgPos = _let->trgPos();
 
  // redistribute source normals 
  PetscInt newSrcLocSize;
  {
    PetscInt dummy;
    VecGetLocalSize(_srcPos, &dummy);
    newSrcLocSize = dummy/3 /*dim*/ ;
  }
  vector<PetscInt>  & newSrcGlobalIndices = _let->newSrcGlobalIndices;
  
  // construct PETSc index set to redistribute source coordinates vector via VecScatter
  // basically, IS is a list of new global indices for local entries of _srcPos
  IS srcNorIS;
  for (size_t i=0; i<newSrcGlobalIndices.size(); i++)
    newSrcGlobalIndices[i] *= 3;
  
  ISCreateBlock(mpiComm(),3, newSrcGlobalIndices.size(),newSrcGlobalIndices.size()?&newSrcGlobalIndices[0]:0,&srcNorIS);
  
  // "newSrcGlobalIndices" might be used later with possibly different block size, so divide it back
  for (size_t i=0; i<newSrcGlobalIndices.size(); i++)
    newSrcGlobalIndices[i] /= 3;
  
  // construct new vector and scatter context to redistribute source coordinates vector via VecScatter
  Vec newSrcNor;
  VecCreateMPI(mpiComm(),3*newSrcLocSize,PETSC_DETERMINE/*global size*/,&newSrcNor);
  
  VecScatter srcNorScatterCtx;
  VecScatterCreate(_srcNor,PETSC_NULL,newSrcNor,srcNorIS, &srcNorScatterCtx);
  ISDestroy(srcNorIS);
  
  // do the actual communication 
  VecScatterBegin(srcNorScatterCtx,_srcNor,newSrcNor,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(srcNorScatterCtx,_srcNor,newSrcNor,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterDestroy(srcNorScatterCtx);
  
  VecDestroy(_srcNor);
  _srcNor = newSrcNor;

  
  /* 2. decide _eq_mm and _mul_mm, and get matmgnt based on that */
  /* See kernel3d_mpi.hpp for what these kernelTypes are */
  switch(_knl.kernelType()) {
	 //laplace kernels
  case KNL_LAP_S_U: _knl_mm = Kernel3d_MPI(KNL_LAP_S_U, _knl.coefs()); break;
  case KNL_LAP_D_U: _knl_mm = Kernel3d_MPI(KNL_LAP_S_U, _knl.coefs()); break;
	 //stokes kernels
  case KNL_STK_S_U: _knl_mm = Kernel3d_MPI(KNL_STK_F_U, _knl.coefs()); break;
  case KNL_STK_S_P: _knl_mm = Kernel3d_MPI(KNL_LAP_S_U, vector<double>()); break;
  case KNL_STK_D_U: _knl_mm = Kernel3d_MPI(KNL_STK_F_U, _knl.coefs()); break;
  case KNL_STK_D_P: _knl_mm = Kernel3d_MPI(KNL_LAP_S_U, vector<double>()); break;
	 //navier kernels
  case KNL_NAV_S_U: _knl_mm = Kernel3d_MPI(KNL_NAV_S_U, _knl.coefs()); break;
  case KNL_NAV_D_U: _knl_mm = Kernel3d_MPI(KNL_NAV_S_U, _knl.coefs()); break;
  default: pA(0);
  }

  /* Get the MatMgnt3d_MPI pointer for kernel _knl_mm and precision np
	* getmmptr creates _matmgnt if it exists or returns a pointer to it */
  _matmgnt  = MatMgnt3d_MPI::getmmptr(_knl_mm, _np);

  /* 3. self setup */
  _nodeVec.resize( _let->nodeVec().size() );
    // ----------------------------------------------------------------------
  /* In the following scope, use the LET to build the global data vectors
	*_glbSrcExaPos, _glbSrcExaNor, _glbSrcExaDen, _glbSrcUpwEquDen
	* These vectors are globally available, so we use VecCreateMPI.
	* See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecCreateMPI.html
	* for more information.
	*/
  {
    if (!mpiRank())
      std::cout<<"Upper. eq. density, number of doubles per octant: "<<datSze(UE)<<endl;

	 //begin
	 //ebiLogInfo( "setup...gdata");
	 /* 1. create contributor vecs */
	 int lclGlbSrcNodeCnt = _let->lclGlbSrcNodeCnt();
	 int lclGlbSrcExaCnt = _let->lclGlbSrcExaCnt();
	 pC( VecCreateMPI(mpiComm(), lclGlbSrcExaCnt*dim(),         PETSC_DETERMINE, &_glbSrcExaPos) );
	 pC( VecCreateMPI(mpiComm(), lclGlbSrcExaCnt*dim(),         PETSC_DETERMINE, &_glbSrcExaNor) );  
	 pC( VecCreateMPI(mpiComm(), lclGlbSrcExaCnt*srcDOF(),        PETSC_DETERMINE, &_glbSrcExaDen) );
	 pC( VecCreateMPI(mpiComm(), lclGlbSrcNodeCnt*datSze(UE),  PETSC_DETERMINE, &_glbSrcUpwEquDen) );
#ifdef DEBUG_LET
	 if (!mpiRank())
	 {
	   PetscInt upwEquGlbSize;
	   VecGetSize(_glbSrcUpwEquDen,&upwEquGlbSize);
	   std::cout<<"Global size of vector of equivalent densities: "<<upwEquGlbSize<<endl;
	 }
#endif
	  

  }
  // ----------------------------------------------------------------------
  /* Build contributor data for a specific processor */
  {
	 //begin
	 //ebiLogInfo("setup...cdata");
	 /* 1. create vecs */
	 int ctbSrcNodeCnt = _let->ctbSrcNodeCnt();
	 int ctbSrcExaCnt = _let->ctbSrcExaCnt();
	 /* Create standard sequential array.  See the following for more information:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecCreateSeq.html
	  * The following creates sequential vectors on each processor for the contributor positions, normals, etc.
	  * based on the sizes of contributor data found when building the LET and figuring out how many nodes this
	  * processor contributes to.
	  */
	 pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcExaCnt*dim(),  &_ctbSrcExaPos) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcExaCnt*dim(),  &_ctbSrcExaNor) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcExaCnt*srcDOF(), &_ctbSrcExaDen) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcNodeCnt*datSze(UE), &_ctbSrcUpwEquDen) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, ctbSrcNodeCnt*datSze(UC), &_ctbSrcUpwChkVal) );
  
	 /* 2. Create scatters
	  * Index Sets are used to setup the vector scatters
	  * See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/index.html
	  * for more information
	  */
	 vector<PetscInt>& ctb2GlbSrcNodeMap = _let->ctb2GlbSrcNodeMap();
	 vector<PetscInt>& ctb2GlbSrcExaMap = _let->ctb2GlbSrcExaMap();
	 /* self index set */
	 IS selfis;
	 /* Combine index set */
	 IS combis;
	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size ctbSrcExaCnt*dim().  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, ctbSrcExaCnt*dim(), 0, 1, &selfis) );
	 for(size_t i=0; i<ctb2GlbSrcExaMap.size(); i++){
		ctb2GlbSrcExaMap[i]*=dim();
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * ctb2GlbSrcExaMap.size() = length of index set
	  * dim() = number of elements in each block
	  * ctb2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, dim(), ctb2GlbSrcExaMap.size(), ctb2GlbSrcExaMap.size()? &ctb2GlbSrcExaMap[0]:0, &combis) );
	 for(size_t i=0; i<ctb2GlbSrcExaMap.size(); i++){
		ctb2GlbSrcExaMap[i]/=dim();
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _ctb2GlbSrcExaPos.  Scatter from shape of _ctbSrcExaPos (specifically indices selfis) to shape of _glbSrcExaPos
	  * (specifically indices combis).
	  */
	 pC( VecScatterCreate(_ctbSrcExaPos, selfis, _glbSrcExaPos, combis, &_ctb2GlbSrcExaPos) );
	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );

	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size ctbSrcExaCnt*srcDOF().  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, ctbSrcExaCnt*srcDOF(), 0, 1, &selfis) );
	 for(size_t i=0; i<ctb2GlbSrcExaMap.size(); i++) {
		ctb2GlbSrcExaMap[i]*=srcDOF();
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * ctb2GlbSrcExaMap.size() = length of index set
	  * srcDOF() = number of elements in each block
	  * ctb2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, srcDOF(),ctb2GlbSrcExaMap.size(), ctb2GlbSrcExaMap.size()? &ctb2GlbSrcExaMap[0]:0, &combis) );
	 for(size_t i=0; i<ctb2GlbSrcExaMap.size(); i++) {
		ctb2GlbSrcExaMap[i]/=srcDOF();
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _ctb2GlbSrcExaDen.  Scatter from shape of _ctbSrcExaDen (specifically indices selfis) to shape of _glbSrcExaDen
	  * (specifically indices combis).  Used in fmm3d_eval_mpi.cpp
	  */
	 pC( VecScatterCreate(_ctbSrcExaDen, selfis, _glbSrcExaDen, combis, &_ctb2GlbSrcExaDen) );
	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );

	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size ctbSrcNodeCnt*datSze(UE).  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, ctbSrcNodeCnt*datSze(UE),0,1,&selfis) );
	 for(size_t i=0; i<ctb2GlbSrcNodeMap.size(); i++){
#ifdef DEBUG_LET
	   assert(ctb2GlbSrcNodeMap[i]>=0);
#endif
	   ctb2GlbSrcNodeMap[i]*=datSze(UE);
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * ctb2GlbSrcNodeMap.size() = length of index set
	  * datSze(UE) = number of elements in each block
	  * ctb2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, datSze(UE), ctb2GlbSrcNodeMap.size(), ctb2GlbSrcNodeMap.size()? &ctb2GlbSrcNodeMap[0]:0, &combis) );
	 for(size_t i=0; i<ctb2GlbSrcNodeMap.size(); i++){
		ctb2GlbSrcNodeMap[i]/=datSze(UE);
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _ctb2GlbSrcUpwEquDen.  Scatter from shape of _ctbSrcUpwEquDen (specifically indices selfis) to shape of _glbSrcUpwEquDen
	  * (specifically indices combis). Used in fmm3d_eval_mpi.cpp
	  */
	 MPI_Barrier(mpiComm());
	 if(!mpiRank())
	   std::cout<<"Creating Ctb2GlbSrcUpwEquDen.... "<<endl;

	 PetscLogEventBegin(Ctb2GlbSctCreate_event,0,0,0,0);
	 pC( VecScatterCreate(_ctbSrcUpwEquDen, selfis, _glbSrcUpwEquDen, combis, &_ctb2GlbSrcUpwEquDen) );
	 PetscLogEventEnd(Ctb2GlbSctCreate_event,0,0,0,0);

	 MPI_Barrier(mpiComm());
	 if(!mpiRank())
	   std::cout<<"Created Ctb2GlbSrcUpwEquDen successfully"<<endl;

	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );
	 
	 /* 3. gather the contributor positions using the pos scatter */
	 PetscScalar zero=0.0;  pC( VecSet(_ctbSrcExaPos, zero) );
	 /* Get the beginning and ending range values this processor is responsible for */
	 PetscInt procLclstart, procLclend; _let->procLclRan(_srcPos, procLclstart, procLclend);
	 double* parr;      pC( VecGetArray(    _srcPos, &parr) );
	 double* narr;      pC( VecGetArray(    _srcNor, &narr) );
	 /* Do an upward tree traversal and store nodes in ordVec */
	 vector<int> ordVec;  pC( _let->upwOrderCollect(ordVec) );
	 /* Traverse nodes */
	 for(size_t i=0; i<ordVec.size(); i++) {
		int gNodeIdx = ordVec[i];
		/* If node has contributor tag for this processor turned on */
		if(_let->node(gNodeIdx).tag() & LET_CBTRNODE) {
		  /* If this node is a terminal/leaf contributor */
		  if(_let->terminal(gNodeIdx)==true) {
			 /* Return pointers to requested matrices */
			 DblNumMat ctbSrcExaPos(this->ctbSrcExaPos(gNodeIdx));
			 DblNumMat ctbSrcExaNor(this->ctbSrcExaNor(gNodeIdx));
			 /* Get the vector of indices for current node's contributor source vector of indices */
			 vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).ctbSrcOwnVecIdxs();
			 for(size_t k=0; k<curVecIdxs.size(); k++) {
				/* Get offset for this processor */
				PetscInt poff = curVecIdxs[k] - procLclstart;
				for(int d=0; d<dim(); d++) {
				  ctbSrcExaPos(d,k) = parr[poff*dim()+d]; /* Contributor exact positions */
				  ctbSrcExaNor(d,k) = narr[poff*dim()+d]; /* Contributor exact normals */
				}
			 }
		  }
		}
	 }
	 pC( VecRestoreArray(_srcPos, &parr) );
	 pC( VecRestoreArray(_srcNor, &narr) );

	 /* Scatter from _ctbSrcExaPos to _glbSrcExaPos using _ctb2GlbSrcExaPos context
	  * See: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterBegin.html
	  * and http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterEnd.html
	  */
	 pC( VecScatterBegin(_ctb2GlbSrcExaPos, _ctbSrcExaPos, _glbSrcExaPos, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterEnd  (_ctb2GlbSrcExaPos, _ctbSrcExaPos, _glbSrcExaPos, INSERT_VALUES, SCATTER_FORWARD) );
	 /* Scatter from _ctbSrcExaNor to _glbSrcExaNor using _ctb2GlbSrcExaPos context */
	 pC( VecScatterBegin(_ctb2GlbSrcExaPos, _ctbSrcExaNor, _glbSrcExaNor, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterEnd  (_ctb2GlbSrcExaPos, _ctbSrcExaNor, _glbSrcExaNor, INSERT_VALUES, SCATTER_FORWARD) );
  } /* Finishes building contributor data */
  
  // ----------------------------------------------------------------------
  /* Build user data for a specific processor */
  {
	 //begin
	 //ebiLogInfo("setup...udata");
	 /* 1. create user vecs */
	 int usrSrcNodeCnt = _let->usrSrcNodeCnt();
	 int usrSrcExaCnt = _let->usrSrcExaCnt();
	 /* Create standard sequential array.  See the following for more information:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecCreateSeq.html
	  * The following creates sequential vectors on each processor for the user positions, normals, etc.
	  * based on the sizes of user data found when building the LET and figuring out how many nodes this
	  * processor contributes to.
	  */
	 pC( VecCreateSeq(PETSC_COMM_SELF, usrSrcExaCnt*dim(),  &_usrSrcExaPos) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, usrSrcExaCnt*dim(),  &_usrSrcExaNor) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, usrSrcExaCnt*srcDOF(), &_usrSrcExaDen) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, usrSrcNodeCnt*datSze(UE), &_usrSrcUpwEquDen) );
	 
	 /* 2. scatter user -> global */
	 vector<PetscInt>& usr2GlbSrcNodeMap = _let->usr2GlbSrcNodeMap();
	 vector<PetscInt>& usr2GlbSrcExaMap = _let->usr2GlbSrcExaMap();
	 /*
	  * Index Sets are used to setup the vector scatters
	  * See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/index.html
	  * for more information
	  */
	 IS selfis;
	 IS combis;
	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size usrSrcExaCnt*dim().  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, usrSrcExaCnt*dim(), 0, 1, &selfis) );
	 for(size_t i=0; i<usr2GlbSrcExaMap.size(); i++){
		usr2GlbSrcExaMap[i]*=dim();
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * usr2GlbSrcExaMap.size() = length of index set
	  * dim() = number of elements in each block
	  * usr2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, dim(), usr2GlbSrcExaMap.size(), usr2GlbSrcExaMap.size()? &usr2GlbSrcExaMap[0]:0, &combis) );
	 for(size_t i=0; i<usr2GlbSrcExaMap.size(); i++){
		usr2GlbSrcExaMap[i]/=dim();
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _usr2GlbSrcExaPos.  Scatter from shape of _usrSrcExaPos (specifically indices selfis) to shape of _glbSrcExaPos
	  * (specifically indices combis).
	  */
	 pC( VecScatterCreate(_usrSrcExaPos, selfis, _glbSrcExaPos, combis, &_usr2GlbSrcExaPos) );
	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );
	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size usrSrcExaCnt*srcDOF().  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, usrSrcExaCnt*srcDOF(), 0, 1, &selfis) );
	 for(size_t i=0; i<usr2GlbSrcExaMap.size(); i++){
		usr2GlbSrcExaMap[i]*=srcDOF();
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * usr2GlbSrcExaMap.size() = length of index set
	  * srcDOF() = number of elements in each block
	  * usr2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, srcDOF(),usr2GlbSrcExaMap.size(), usr2GlbSrcExaMap.size()? &usr2GlbSrcExaMap[0]:0, &combis) );
	 for(size_t i=0; i<usr2GlbSrcExaMap.size(); i++){
		usr2GlbSrcExaMap[i]/=srcDOF();
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _usr2GlbSrcExaDen.  Scatter from shape of _usrSrcExaDen (specifically indices selfis) to shape of _usrSrcExaDen
	  * (specifically indices combis).
	  */
	 pC( VecScatterCreate(_usrSrcExaDen, selfis, _glbSrcExaDen, combis, &_usr2GlbSrcExaDen) );
	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );

	 /* Creates a data structure for an index set containing a list of evenly spaced integers.
	  * Starts at 0 and increments by 1 until size usrSrcNodeCnt*datSze(UE).  Store in selfis.  See:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateStride.html
	  */
	 pC( ISCreateStride(PETSC_COMM_SELF, usrSrcNodeCnt*datSze(UE),0,1,&selfis) );
	 for(size_t i=0; i<usr2GlbSrcNodeMap.size(); i++){
		usr2GlbSrcNodeMap[i]*=datSze(UE);
	 }
	 /* Creates a data structure for an index set containing a list of integers. The indices are relative to entries, not blocks
	  * usr2GlbSrcNodeMap.size() = length of index set
	  * datSze(UE) = number of elements in each block
	  * usr2GlbSrcExaMap = list of integers
	  * combis = new index set
	  * See for more info: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/IS/ISCreateBlock.html
	  */
	 pC( ISCreateBlock( PETSC_COMM_SELF, datSze(UE), usr2GlbSrcNodeMap.size(), usr2GlbSrcNodeMap.size()? &usr2GlbSrcNodeMap[0]:0, &combis) );
	 for(size_t i=0; i<usr2GlbSrcNodeMap.size(); i++){
		usr2GlbSrcNodeMap[i]/=datSze(UE);
	 }
	 /* http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreate.html
	  * Create new scatter context _usr2GlbSrcUpwEquDen.  Scatter from shape of _usrSrcUpwEquDen (specifically indices selfis) to shape of _glbSrcUpwEquDen
	  * (specifically indices combis). Used in fmm3d_eval_mpi.cpp
	  */
	 
	 // memory usage profiling
	 {
	   PetscLogDouble mem1, mem2, locScatUsage, maxScatUsage, minScatUsage;
	   PetscMallocGetCurrentUsage(&mem1);

	   MPI_Barrier(mpiComm());
	   if(!mpiRank())
	     std::cout<<"Creating usr2GlbSrcUpwEquDen.... "<<endl;
	   PetscLogEventBegin(Usr2GlbSctCreate_event,0,0,0,0);
	   pC( VecScatterCreate(_usrSrcUpwEquDen, selfis, _glbSrcUpwEquDen, combis, &_usr2GlbSrcUpwEquDen) );
	   PetscLogEventEnd(Usr2GlbSctCreate_event,0,0,0,0);
	   MPI_Barrier(mpiComm());
	   if(!mpiRank())
	     std::cout<<"Created usr2GlbSrcUpwEquDen successfully"<<endl;

	   PetscMallocGetCurrentUsage(&mem2);
	   locScatUsage = mem2-mem1;
	   MPI_Reduce ( &locScatUsage, &maxScatUsage, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, 0, mpiComm() );
	   MPI_Reduce ( &locScatUsage, &minScatUsage, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, 0, mpiComm() );
	   if(!mpiRank())
	     std::cout<<"fmm setup: memory used by scatter context 'upper equiv. dens. users-to-owners', min="<<minScatUsage<<" max="<<maxScatUsage<<endl;
	 } 

	 pC( ISDestroy(selfis) );
	 pC( ISDestroy(combis) );
	 
	 /* 3. do vecscatter */
	 /* Scatter from _usrSrcExaPos to _usrSrcExaPos using _usr2GlbSrcExaPos context
	  * See: 
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterBegin.html
	  * and http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterEnd.html
	  */
	 pC( VecScatterBegin(_usr2GlbSrcExaPos, _glbSrcExaPos, _usrSrcExaPos, INSERT_VALUES, SCATTER_REVERSE) );
	 pC( VecScatterEnd  (_usr2GlbSrcExaPos, _glbSrcExaPos, _usrSrcExaPos, INSERT_VALUES, SCATTER_REVERSE) );
	 /* Scatter from _usrSrcExaNor to _glbSrcExaNor using _usr2GlbSrcExaPos context */
	 pC( VecScatterBegin(_usr2GlbSrcExaPos, _glbSrcExaNor, _usrSrcExaNor, INSERT_VALUES, SCATTER_REVERSE) );
	 pC( VecScatterEnd  (_usr2GlbSrcExaPos, _glbSrcExaNor, _usrSrcExaNor, INSERT_VALUES, SCATTER_REVERSE) );

	 /* Do an upward node traversal collection */
	 vector<int> ordVec; pC( _let->upwOrderCollect(ordVec) );
	 /* Traverse node collection */
	 for(size_t i=0; i<ordVec.size(); i++) {
		int gNodeIdx = ordVec[i];
		/* If a node has the evaluator tag turned on for this processor */
		if(_let->node(gNodeIdx).tag() & LET_EVTRNODE) {
		  /* For each node in current node's v-list increment Number of boxes which have this node in their V-lists */
		  for(vector<int>::iterator vi=_let->node(gNodeIdx).Vnodes().begin(); vi!=_let->node(gNodeIdx).Vnodes().end(); vi++) {
			 node(*vi).vLstOthNum() ++;
		  }
		}
	 }
  } /* End of building user data for processor */
  // ----------------------------------------------------------------------
  
  { /* Build evaluator data */
	 //begin
	 //ebiLogInfo("setup...edata");
	 /* 1. create vecs */
	 int evaTrgNodeCnt = _let->evaTrgNodeCnt();
	 int evaTrgExaCnt = _let->evaTrgExaCnt();
	 /* Create standard sequential array.  See the following for more information:
	  * http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecCreateSeq.html
	  * The following creates sequential vectors on each processor for the evaluator positions, normals, etc.
	  */
	 pC( VecCreateSeq(PETSC_COMM_SELF, evaTrgExaCnt*dim(),  &_evaTrgExaPos) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, evaTrgExaCnt*trgDOF(), &_evaTrgExaVal) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, evaTrgNodeCnt*datSze(DE), &_evaTrgDwnEquDen) );
	 pC( VecCreateSeq(PETSC_COMM_SELF, evaTrgNodeCnt*datSze(DC), &_evaTrgDwnChkVal) );
	 
	 /* 2. gather data from _trgPos (target positions) */
	 /* Get start and end of the local range for this processor */
	 PetscInt procLclStart, procLclEnd; _let->procLclRan(_trgPos, procLclStart, procLclEnd);
	 double* parr; pC( VecGetArray(_trgPos, &parr) );
	 /* Do an upward node traversal collection */
	 vector<int> ordVec; pC( _let->upwOrderCollect(ordVec) );
	 /* Traverse node collection */
	 for(size_t i=0; i<ordVec.size(); i++) {
		int gNodeIdx = ordVec[i];
		/* If this node has evaluate tag turned on */
		if(_let->node(gNodeIdx).tag() & LET_EVTRNODE) {
		  /* And if it is a leaf/terminal */
		  if(_let->terminal(gNodeIdx)==true) {
			 DblNumMat evaTrgExaPos(this->evaTrgExaPos(gNodeIdx));
			 /* Look at current node's evaluatir target vector of indices */
			 vector<PetscInt>& curVecIdxs = _let->node(gNodeIdx).evaTrgOwnVecIdxs();
			 for(size_t k=0; k<curVecIdxs.size(); k++) {
				PetscInt poff = curVecIdxs[k]-procLclStart;
				for(int d=0; d<dim(); d++)
				  evaTrgExaPos(d,k) = parr[poff*dim()+d]; /* Evaluator exact target positions matrix */
			 }
		  }
		}
	 }
	 pC( VecRestoreArray(_trgPos, &parr) );
	 /* Count the number of nodes in each node's v-list */
	 for(size_t i=0; i<ordVec.size(); i++) {
		int gNodeIdx = ordVec[i];
		if( _let->node(gNodeIdx).tag() & LET_EVTRNODE) {
		  node(gNodeIdx).vLstInNum() = _let->node(gNodeIdx).Vnodes().size();
		}
	 }
  } /* End of build evaluator data */
    
  return(0);
}


