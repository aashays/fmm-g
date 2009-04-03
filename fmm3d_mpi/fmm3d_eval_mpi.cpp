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
using std::cout;
using std::endl;

// ---------------------------------------------------------------------- 
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::InsLayerInterSectsRange"
bool FMM3d_MPI::ParInsLayerIntersectsRange(ot::TreeNode oct, int r1, int r2)
{
  ot::TreeNode parent = oct.getParent();
  ot::TreeNode root_oct(3/*dim*/, _let->maxLevel());

  vector<ot::TreeNode> neighbs = parent.getAllNeighbours();
  neighbs.push_back(parent); // we add parent itself to get complete insulation layer

  // in the loop below we find min and max neighbs (min/max in morton sense)
  // we also substitute non-existent neighbs (which are returned as root octants)
  // with the parent itself
  ot::TreeNode min_neighb=parent;
  ot::TreeNode max_neighb=parent;
  for (size_t i=0; i<neighbs.size(); i++)
  {
    if (neighbs[i]==root_oct)  //this  means corresponding neighb. does not exist
      continue;

    if (neighbs[i].getDLD()>=_let->procMins[r1]  && (r2==mpiSize()-1 || neighbs[i].getDFD()<_let->procMins[r2+1] ))
      return true;
    // 	  min_neighb=min(min_neighb,neighbs[i]);
    // 	  max_neighb=max(max_neighb,neighbs[i]);
  }

      // check if insulation layer intersects with the area controlled by processors r1 ... r2
//       bool overlaps = (max_neighb.getDLD()>=_let->procMins[r1]  && (r2==mpiSize()-1 || min_neighb.getDFD()<_let->procMins[r2+1] ));
//       if (oct.getX()==0 && oct.getY()==805306368 && oct.getZ()==268435456 && oct.getLevel()==3)
// 	cout<<"Rank: "<<mpiRank()<<" Octant: "<< oct << " r1="<<r1<<" r2="<<r2<<" Overlaps="<<overlaps<<endl;
//       return overlaps;
  return false;  // if we got to this point, none of the neighbs intersects with the range
}

#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::ExchangeOctantsTreeBased"
int FMM3d_MPI::ExchangeOctantsTreeBased()
{
  using namespace std;
  // so far we only support size of communicator which is power of 2
  pA( !(mpiSize() & (mpiSize()-1)) );
  vector<Let3d_MPI::Node> & nodes = _let->nodeVec();

  vector<ot::TreeNode> * l; // l contains ``current list of octants'' at each iteration;  initially l contains local shared octants; finally l contains all necessary octants from other processors
  l = new vector<ot::TreeNode> ; 

  // make a list of ``local shared octants'';  that is octants that this process owns and which (octants) have insulation layer intersecting with areas controlled by other processors
  int q=0;  //  0 means root node;   
  while(q!=-1) 
  {
    // if node q is not a leaf
    // and the span of the insulation layer of this node is not entirely local
    // then  push all <children of q THAT WE OWN> to all processors whos areas intersect with the 
    // insulation layer
    // and go to first child of q and repeat
    if (!_let->terminal(q))
    {
      unsigned q_lev=_let->depth(q);
      unsigned x,y,z;
      x = (nodes[q].path2Node())(0) << _let->maxLevel()-q_lev;
      y = (nodes[q].path2Node())(1) << _let->maxLevel()-q_lev;
      z = (nodes[q].path2Node())(2) << _let->maxLevel()-q_lev;

      ot::TreeNode oct (x,y,z,q_lev,3/*dim*/,_let->maxLevel());
      ot::TreeNode root_oct(3/*dim*/, _let->maxLevel());

      vector<ot::TreeNode> neighbs = oct.getAllNeighbours();
      neighbs.push_back(oct); // we add octant itself to get complete insulation layer

      // in the loop below we find min and max neighbs (min/max in morton sense)
      // we also substitute non-existent neighbs (which are returned as root octants)
      // with the octant itself ("oct")
      ot::TreeNode min_neighb=oct;
      ot::TreeNode max_neighb=oct;
      for (size_t i=0; i<neighbs.size(); i++)
      {
	if (neighbs[i]==root_oct)  //this  means corresponding neighb. does not exist
	  neighbs[i]=oct;
	else
	{
	  min_neighb=min(min_neighb,neighbs[i]);
	  max_neighb=max(max_neighb,neighbs[i]);
	}
      }

      // find the lower bound for processes that can intersect with the insulation layer
      // that is, no processor less that first_cpu can intersect
      // we use getDFD() since min_neighb may intersect domains of many processors
      vector<ot::TreeNode>::const_iterator first_cpu = upper_bound(_let->procMins.begin(),_let->procMins.end(),min_neighb.getDFD());
      first_cpu--; //we actually need the last element which is not greater than search key

      // past_last_cpu is upper bound: any processor with rank>=past_last_cpu does not intersect with insul. layer 
      // we use getDLD() since max_neighb may intersect domains of many processors
      vector<ot::TreeNode>::const_iterator past_last_cpu = upper_bound(_let->procMins.begin(),_let->procMins.end(),max_neighb.getDLD());
      // now past_last_cpu points to first element which is greater than search key (maybe procMins.end() )

      if (past_last_cpu-first_cpu > 1)  // if insulation layer is not entirely local...
      {
	// if insulation layer of an octant lies inside the area owned by single process, and this octant encloses some sources that our process owns, then this "single process" is OUR process
	// mark all children (which must exist, since node is non-terminal) THAT WE CONTRIBUTE TO
	// as ``shared''
	for (int chd_num=0; chd_num<8; chd_num++)
	{
	  int chd_index=nodes[q].chd()+chd_num;
	  if (nodes[chd_index].tag() & LET_CBTRNODE)
	  {
	    unsigned lev=_let->depth(chd_index);
	    unsigned x,y,z;
	    x = (nodes[chd_index].path2Node())(0) << _let->maxLevel()-lev;
	    y = (nodes[chd_index].path2Node())(1) << _let->maxLevel()-lev;
	    z = (nodes[chd_index].path2Node())(2) << _let->maxLevel()-lev;

	    ot::TreeNode oct (x,y,z,lev,3/*dim*/,_let->maxLevel());
	    oct.setWeight(chd_index); // we will sort octants and we need to keep track of  original indices
	    l->push_back(oct);
	  }
	}
	// go to first child of q 
	q=nodes[q].chd();
	continue;  // (while loop)
      }
    } // end if (!terminal(q))

    // otherwise (i.e. if q is a leaf, or does not enclose any sources, or its insulation 
    // layer is  entirely local) go to "next" node 
    // next node is next sibling (if we are not last sibling or root), or next sibling of a parent 
    // (if parent is not last sibling itself) and so on; if we at the last node, just exit
    do
    {
      int p=nodes[q].par();
      if (p==-1)
      {
	// q is root octant, thus there are no more nodes to process
	q=-1; // exit flag
	break;
      }
      if (q - nodes[p].chd() < 7 )   // if q is not the last child ... (there are 8 children overall)
      {
	q++; // go to next sibling
	break;
      }
      else
	q=p; // go to parent and see what sibling parent is
    }
    while (true);

  } // end while(q!=-1)

  sort(l->begin(), l->end());

  // load partial densities
  int density_size = _matmgnt->plnDatSze(UE);
  vector<double> * densities = new vector<double>(density_size*l->size(),10);
  for (size_t i=0; i<l->size(); i++)
  {
    double * data = ctbSrcUpwEquDen( (*l)[i].getWeight() )._data;
    for (int j=0; j<density_size; j++)
      (*densities)[i*density_size+j] = data[j];
  }


  MPI_Datatype mpi_treenode;
  MPI_Type_contiguous( sizeof(ot::TreeNode), MPI_BYTE, &mpi_treenode);
  MPI_Type_commit(&mpi_treenode);

  // communication loop:
  for (int two_power=mpiSize()/2; two_power>0; two_power>>=1)
  {
    vector<ot::TreeNode>  & ll = *l;
    int partner = mpiRank() ^ two_power;
    int r1 = partner & (mpiSize()-two_power);
    int r2 = partner | (two_power-1);
    int q1 = mpiRank() & (mpiSize()-two_power);
    int q2 = mpiRank() | (two_power-1);

//     if (!mpiRank())
//       std::cout<<"Bit: "<<two_power<<endl;
//     MPI_Barrier(mpiComm());
//     std::cout<<"Rank: "<<mpiRank()<<" peer: "<<partner<<endl;
//     MPI_Barrier(mpiComm());

    // build the list of octants to send to partner
    vector<ot::TreeNode> to_send;
    for(size_t q=0; q<ll.size(); q++)
    {
      // check if insulation layer intersects with the area controlled by processors r1 ... r2
      if (ParInsLayerIntersectsRange(ll[q],r1,r2))
      {
	to_send.push_back(ll[q]);
	to_send.back().setWeight(q);  // to keep track of density
      }
    }

    // do send and receive
    // first do sizes
    int send_size =  to_send.size();
    int recv_size;
    MPI_Status status;  // we don't really use this
    MPI_Sendrecv(&send_size, 1, MPI_INT, partner, 0, &recv_size, 1, MPI_INT, partner, 0, mpiComm(), &status);

    vector<ot::TreeNode> recvd(recv_size);
    //  int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status) 
    MPI_Sendrecv(send_size? &to_send[0]:0, send_size, mpi_treenode, partner, 0, recv_size? &recvd[0]:0, recv_size, mpi_treenode, partner, 0, mpiComm(), &status);
    // now recvd  contains octants received from partner

    // group densities-to-send together and allocate space for received densities
    vector<double> densities_to_send(send_size*density_size);
    for (int i=0; i<send_size; i++)
    {
      double * data = &(*densities)[0] + density_size*to_send[i].getWeight();
      for (int j=0; j<density_size; j++)
	densities_to_send[i*density_size+j] = data[j];
    }
    vector<double> recvd_densities(recv_size*density_size);

    // exchange densities
    MPI_Sendrecv(send_size? &densities_to_send[0]:0, send_size*density_size, MPI_DOUBLE, partner, 0, recv_size? &recvd_densities[0]:0, recv_size*density_size, MPI_DOUBLE, partner, 0, mpiComm(), &status);

    // merge l and received octants; take only those octants from l, which are adressed to this process or still have to be sent somewhere; we assume both l and recvd  are Morton sorted (in ascending order)
    vector<ot::TreeNode> * new_l = new vector<ot::TreeNode>;
    vector<double> * new_densities = new vector<double>;
    new_densities->reserve(densities->size()+recvd_densities.size()); // we'll resize it eventually if necessary

    // merging loop, we assume both "l" and "recvd" are Morton-sorted 
    size_t ll_ptr = 0;
    size_t recv_ptr = 0;
    while(ll_ptr<ll.size() && recv_ptr<recvd.size())
    {
      if (ll[ll_ptr] < recvd[recv_ptr])
      {
	if (ParInsLayerIntersectsRange(ll[ll_ptr],q1,q2))   // octants from recvd should automatically satisfy this
	{
	  new_l->push_back(ll[ll_ptr]); 
	  for (int i=0; i<density_size; i++)
	    new_densities->push_back( (*densities)[ll_ptr*density_size+i] );
	}
	ll_ptr++;
      }
      else
      {
	new_l->push_back(recvd[recv_ptr]); 
	if (ll[ll_ptr] == recvd[recv_ptr])
	{
	  for (int i=0; i<density_size; i++)
	    new_densities->push_back( (*densities)[ll_ptr*density_size+i]  + recvd_densities[recv_ptr*density_size+i] );
	  ll_ptr++;
	}
	else
	  for (int i=0; i<density_size; i++)
	    new_densities->push_back( recvd_densities[recv_ptr*density_size+i] );
	recv_ptr++;
      }
    }

    while(ll_ptr<ll.size())
    {
      if (ParInsLayerIntersectsRange(ll[ll_ptr],q1,q2))   // octants from recvd should automatically satisfy this
      {
	new_l->push_back(ll[ll_ptr]); 
	for (int i=0; i<density_size; i++)
	  new_densities->push_back( (*densities)[ll_ptr*density_size+i] );
      }
      ll_ptr++;
    }

    while(recv_ptr<recvd.size())
    {
      new_l->push_back(recvd[recv_ptr]); 
      for (int i=0; i<density_size; i++)
	new_densities->push_back( recvd_densities[recv_ptr*density_size+i] );
      recv_ptr++;
    }

    delete l;
    l=new_l;
    delete densities;
    densities = new_densities;

  } // end of the communication loop

  MPI_Type_free(&mpi_treenode);


  // now insert received octants in local tree
  // some received octants may already be present in the tree, then just set the global indices
  // for now, we'll do simplistic implementation:  for each  received octant we start from root and go down the tree to find an appropriate place
  for (size_t i=0; i<l->size(); i++)
  {
    int level = (*l)[i].getLevel();
    vector<int> coord(3);
    coord[0] = ((*l)[i].getX())>>( _let->maxLevel()-level );
    coord[1] = ((*l)[i].getY())>>( _let->maxLevel()-level );
    coord[2] = ((*l)[i].getZ())>>( _let->maxLevel()-level );

    int q= 0;  // we start from root octant

    while(true)
    {
#ifdef DEBUG_LET
      assert(nodes[q].depth() <= level);
#endif
      if (nodes[q].depth() == level)
      {
	// at this point "nodes[q]" should be same octant as "octData"
#ifdef DEBUG_LET
	Index3 path = nodes[q].path2Node();
	for(int j=0; j<3; j++)
	  assert(path[j]==coord[j]);
#endif
	//  if this octant is really used on this process (is on some list of some "local" octant)
	if ( nodes[q].tag() & LET_USERNODE )
	{
	  double * data = usrSrcUpwEquDen(q)._data;
	  for (int j=0; j<density_size; j++)
	    data[j]= (*densities)[i*density_size+j];
	}
	break;
      }
      else // we must go deeper
      {
	// first, if nodes[q] is leaf, complain fiercely
	if (nodes[q].chd()==-1)
	{
	  cout<<"Unneded octant! My rank: "<<mpiRank()<<" Octant: "<< (*l)[i] << endl;
	  assert(false);
	}
	
	// now go to the appropriate child of nodes[q] (maybe just created by code above)
	unsigned idx[3];
	for(int j=0; j<3; j++)
	  idx[j]=( coord[j] >> (level - nodes[q].depth() - 1) ) & 1; 
	q = nodes[q].chd() + idx[0]*4+idx[1]*2+idx[2];
      }
    }
  }
  delete l;
  delete densities;
  return 0;
}

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
      
  PetscTruth use_treebased_broadcast;
  PetscOptionsHasName(0,"-use_treebased_broadcast",&use_treebased_broadcast);

  //1. zero out vecs.  This includes all global, contributor, user, evaluator vectors.
  PetscScalar zero=0.0;
  pC( VecSet(trgVal, zero) );
  pC( VecSet(_glbSrcExaDen, zero) );
  if (!use_treebased_broadcast)
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

#ifdef COMPILE_GPU
  upComp_t *UpC;
#endif

  PetscTruth gpu_s2m;
  PetscOptionsHasName(0,"-gpu_s2m",&gpu_s2m);
  if (gpu_s2m)
    // compute s2m for all leaves at once
  {
#ifdef COMPILE_GPU
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
#else
    SETERRQ(1,"GPU code not compiled");
#endif
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
#ifdef COMPILE_GPU
	    for (int j = 0; j < ctbSrcUpwChkValgNodeIdx.m(); j++) 
	      ctbSrcUpwChkValgNodeIdx(j) = UpC->trgVal[gNodeIdx][j];
#else
	    SETERRQ(1,"GPU code not compiled");
#endif
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
#ifdef COMPILE_GPU
    free (UpC->src_);
    free (UpC->srcBoxSize);
    free (UpC->trgCtr);
    free (UpC->trgRad);
    for (int i = 0; i < ordVec.size(); i++)
      free (UpC->trgVal[ordVec[i]]);
    free (UpC->trgVal);
    free (UpC);
#else
    SETERRQ(1,"GPU code not compiled");
#endif
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
    if (!use_treebased_broadcast)
    {
      PetscLogEventBegin(EvalCtb2GlbEqu_event,0,0,0,0);
      pC( VecScatterBegin( _ctb2GlbSrcUpwEquDen, _ctbSrcUpwEquDen, _glbSrcUpwEquDen,    ADD_VALUES, SCATTER_FORWARD) );
      pC( VecScatterEnd(_ctb2GlbSrcUpwEquDen,   _ctbSrcUpwEquDen, _glbSrcUpwEquDen,    ADD_VALUES, SCATTER_FORWARD) );
      PetscLogEventEnd(EvalCtb2GlbEqu_event,0,0,0,0);
    }

    MPI_Barrier(mpiComm()); // for accurate timing, since synchronization is possible inside VecScatterBegin/End
    PetscLogEventBegin(EvalGlb2UsrEquBeg_event,0,0,0,0);
    // sending equiv. densities from owners to users is overlapped (in some cases) with U-list computations
    if(use_treebased_broadcast)
    {
      if (!mpiRank())
	cout<<"Using tree-based broadcast"<<endl;
      ExchangeOctantsTreeBased();
    }
    else
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
#ifdef COMPILE_GPU
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
#endif
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

  if (!skip_communication && !use_treebased_broadcast)
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
		  DblNumVec evaTrgExaVal_gNodeIdx(evaTrgExaVal(gNodeIdx));
		  pC( DwnEqu2TrgChk_dgemv(_let->center(gNodeIdx), _let->radius(gNodeIdx), evaTrgExaPos(gNodeIdx), evaTrgDwnEquDen(gNodeIdx), evaTrgExaVal_gNodeIdx) );
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



