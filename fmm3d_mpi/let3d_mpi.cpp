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

#include <cstring>
#include "let3d_mpi.hpp"
#include "TreeNode.h"
#include "externVars.h"
#include "mpi_workarounds.hpp"
#include "manage_petsc_events.hpp"
#include "parUtils.h"

using std::min;
using std::max;
using std::set;
using std::queue;
using std::ofstream;
using std::cout;
using std::endl;
using std::vector;

//-----------------------------------------------------------------------
Let3d_MPI::Let3d_MPI(const string& p):
  ComObject_MPI(p), _srcPos(NULL), _trgPos(NULL), _ctr(0.0), _rootLevel(0),
  _ptsMax(150), _maxLevel(10)
{
}

Let3d_MPI::~Let3d_MPI()
{
}

#undef __FUNCT__
#define __FUNCT__ "CreateMins"
void Let3d_MPI::CreateMins(const vector<ot::TreeNode> & linOct)
{
  ot::TreeNode sendMin;
  int iAmActive;

  if(!linOct.empty()) {
    sendMin = linOct[0]; //local min
    iAmActive=1;
  }else {
    sendMin = ot::TreeNode (3/*dim*/ ,maxLevel());  //push some junk (e.g., root octant) for inactive nodes
    iAmActive=0;
  }

  int activeSize;
  MPI_Allreduce(&iAmActive,&activeSize,1,MPI_INT,MPI_SUM,mpiComm());

  procMins.resize(mpiSize());
#ifdef USE_ALLGATHER_FIX
  {
    if(!mpiRank())
      cout<<"LET setup: using AllToAll instead of AllGather"<<endl;

    vector<ot::TreeNode> replicas(mpiSize(), sendMin);
    MPI_Alltoall(
	&replicas[0], 
	1, 
	par::Mpi_datatype<ot::TreeNode >::value(), 
	&procMins[0], 
	1, 
	par::Mpi_datatype<ot::TreeNode >::value(), 
	mpiComm() );
  }
#else
  MPI_Allgather(
      &sendMin, 
      1,
      par::Mpi_datatype<ot::TreeNode >::value(),
      &procMins[0], 
      1,
      par::Mpi_datatype<ot::TreeNode >::value(),
      mpiComm() );
#endif
  
  procMins.resize(activeSize);
}

#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::calcLeafWeights"
int Let3d_MPI::calcLeafWeights(vector<ot::TreeNode> & localLeaves)
{
  // get the weights from options
  PetscTruth flg;
  PetscInt Uwt, Vwt, Wwt, Xwt;
  PetscOptionsGetInt(0, "-u_weight", &Uwt, &flg); pA(flg);
  PetscOptionsGetInt(0, "-v_weight", &Vwt, &flg); pA(flg);
  PetscOptionsGetInt(0, "-w_weight", &Wwt, &flg); pA(flg);
  PetscOptionsGetInt(0, "-x_weight", &Xwt, &flg); pA(flg);

  PetscInt np;
  PetscOptionsGetInt(0, "-fmm3d_np",     &np,     &flg); pA(flg);

  np = np*np*np - (np-2)*(np-2)*(np-2);  // number of points on the equivalent surface

  // find the first local leaf in LET
  // loop simultaneously over leaves in LET and leaves in linOct (in both cases loop in morton order)
  // for each leaf in LET, calculate weight and store it in linOct[i]

  // p is an index used to go around the tree; 
  // after processing of each leaf, p must contain index of its parent 
  // p initially contains index of root
  int p = 0;
  unsigned p_lev;  // level of p (0 corresponds to root)

  for(size_t n=0; n<localLeaves.size(); n++)
  {
    ot::TreeNode oct = localLeaves[n];
    unsigned x=oct.getX();
    unsigned y=oct.getY();
    unsigned z=oct.getZ();
    unsigned lev=oct.getLevel();

    //look for nearest common ancestor of last processed leaf (if any) and oct
    while(true) 
    {
#ifdef DEBUG_LET
      assert(p>=0);
#endif
      p_lev=_nodeVec[p].depth();

      // translate integer coordinates to DENDRO format
      unsigned tmpX = (_nodeVec[p].path2Node())(0)<<(maxLevel()-p_lev);
      unsigned tmpY = (_nodeVec[p].path2Node())(1)<<(maxLevel()-p_lev);
      unsigned tmpZ = (_nodeVec[p].path2Node())(2)<<(maxLevel()-p_lev);

      ot::TreeNode tmpOctant = ot::TreeNode(tmpX,tmpY,tmpZ,p_lev,3,maxLevel());
      if (tmpOctant.isAncestor(oct))
	break;
      else
	p=_nodeVec[p].par();
    }

    // now localLeaves[n] must be a descendant of _nodeVec[p]
    while ( (p_lev=_nodeVec[p].depth()) < lev )
    {
      // we need to add children to some child of p
      // first find out to which one
      Index3 idx;

#ifdef DEBUG_LET
      assert(lev>p_lev);
#endif
      idx(0)=(x>>(maxLevel()-p_lev-1)) & 1; 
      idx(1)=(y>>(maxLevel()-p_lev-1)) & 1; 
      idx(2)=(z>>(maxLevel()-p_lev-1)) & 1; 

      p = child(p, idx);
    }

    // debug: check if integer coordinates of oct and p coincide  (up to format difference)
#ifdef DEBUG_LET
    assert(_nodeVec[p].chd()==-1);
    assert(oct.getX()==(unsigned)(_nodeVec[p].path2Node())(0)<<(maxLevel()-lev));
    assert(oct.getY()==(unsigned)(_nodeVec[p].path2Node())(1)<<(maxLevel()-lev));
    assert(oct.getZ()==(unsigned)(_nodeVec[p].path2Node())(2)<<(maxLevel()-lev));
#endif

    // ---- calculate the weight ----
    {
      unsigned long wt=0;
      Node & nod = _nodeVec[p];
      if( nod.tag() & LET_EVTRNODE) 
      {
	// U-list
	for(size_t i=0; i< nod.Unodes().size(); i++)
	{
	  Node & nod2 =_nodeVec[  nod.Unodes()[i]  ];
	  wt += Uwt*nod.evaTrgExaNum()*nod2.glbSrcExaNum();
	}

	// V-list
	wt += (unsigned long)(Vwt*nod.Vnodes().size()*np*sqrt(double(np))); // translation complexity for single box on V-list is of order (np)^(3/2)

	// W-list
	for(size_t i=0; i< nod.Wnodes().size(); i++)
	{
	  Node & nod2 =_nodeVec[  nod.Wnodes()[i]  ];
	  if (nod2.glbSrcExaNum()> np || nod2.glbSrcExaNum()<0) // if nod2 is a leaf and contains more than "np" sources OR nod2 is not a leaf at all
	    wt += Wwt*nod.evaTrgExaNum()*np;
	  else
	    wt += Wwt*nod.evaTrgExaNum()*nod2.glbSrcExaNum();
	}

	// X-list
	int tmp=min(nod.evaTrgExaNum(),int(np));
	for(size_t i=0; i< nod.Xnodes().size(); i++)
	{
	  Node & nod2 =_nodeVec[  nod.Xnodes()[i]  ];
	  wt += Xwt*tmp*nod2.glbSrcExaNum();
	}
      }
      if (wt<=UINT_MAX)
	localLeaves[n].setWeight(wt);
      else
	SETERRQ(1, "some leaf has weight greater than UINT_MAX");
    }

    // go back to parent of current  leaf, to prepare for next iteration
    p=_nodeVec[p].par();
  }
  return 0;
}

// ----------------------------------------------------------------------
/* Setup builds the basic structures needed for processing source data and building target data */
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::setup"
int Let3d_MPI::setup()
{
  /* Get the maximum number of points per leaf level box and max number of levels in the tree */
  PetscTruth flg = PETSC_FALSE;
  PetscInt ptsMaxTemp;
  PetscInt maxLevelTemp;
  pC( PetscOptionsGetInt( prefix().c_str(), "-ptsMax",   &ptsMaxTemp,   &flg) ); pA(flg==true);
  pC( PetscOptionsGetInt( prefix().c_str(), "-maxLevel", &maxLevelTemp, &flg) ); pA(flg==true);
  _ptsMax = (int)ptsMaxTemp;
  _maxLevel = (int)maxLevelTemp;

#ifdef DEBUG_LET
  if(!mpiRank())
    cout<<"Compiled with DEBUG_LET defined"<<endl;
#endif
  
  // ------- create finest-level octree using DENDRO code ---------
  vector<ot::TreeNode> linOct;
  PetscInt regLevel;
  PetscTruth opt_found;
  PetscOptionsGetInt(PETSC_NULL,"-useRegularOctreeAtLevel", &regLevel,&opt_found);

  if (opt_found)
  {
    if (!mpiRank())
      cout<<"Creating regular octree at level "<<regLevel<<endl;

    createRegularOctree(linOct,regLevel,3/*dim*/,maxLevel(),mpiComm());
  }
  else   // create octree based on points ...
  {
    // copying below can be avoided if one modifies DENDRO code
    size_t localSrcSize =  procLclNum(_srcPos)*3 /*dim*/ ;
    vector<double> pts(localSrcSize);
    double* srcPosarr; 
    pC( VecGetArray(_srcPos, &srcPosarr) );
    for(size_t k=0; k<localSrcSize; k++)
      pts[k]=srcPosarr[k];
    pC( VecRestoreArray(_srcPos, &srcPosarr) );

    double gSize[3]={1.0, 1.0, 1.0};
    
    // pts is cleared inside points2octree
    ot::points2Octree(pts, gSize, linOct, 3/*dim*/, maxLevel()/*maxDepth*/, ptsMax(), mpiComm());

    // balance finest octree if asked to do so
    PetscOptionsHasName(0,"-balance_octree",&opt_found);
    if(opt_found)
    {
      if (!mpiRank())
	cout<<"Balancing finest octree"<<endl;
      vector<ot::TreeNode> balOct;
      ot::balanceOctree (linOct, balOct, 3/*dim*/, maxLevel(), 1/*incCorner*/, mpiComm(), NULL, NULL);
      linOct=balOct; // dumb copy, but should not be too expensive
    }
  }

  // handle degenerate case when there is only one octant (root) in a tree -- split root octant into 8 children
  if (linOct.size()==1 && linOct[0]==ot::TreeNode (3/*dim*/, maxLevel()))
  {
    assert(0==mpiRank());  // this process must be first one
    linOct.clear();
    ot::TreeNode(3/*dim*/, maxLevel()).addChildren(linOct);
  }

  // now repartition the octree, if asked to do so; so far repartitioning is naive (based on number of leaves in each block)
  // note that points2octree itself calls blockpart with the weights proportional (?) to number of points
  PetscTruth repartition;
  char repartition_type[50];
  PetscOptionsGetString(0,"-repartition",repartition_type,50,&repartition);
  if (repartition)
    if (0==strcmp(repartition_type,"blockPart"))
    {
      if (!mpiRank())
	cout<<"Repartitioning finest octree using blockPart"<<endl;
      // some processors might have zero octants in linOct and should not, e.g., call BlockPart 
      // create communicator for those processes which  have linOct.size()>0
      // relative order of processes will be preserved 
      MPI_Comm commActive;
      MPI_Comm_split(mpiComm(), linOct.size()>0? 1:MPI_UNDEFINED, 1/*key*/, &commActive);

      bool iAmActive=linOct.size()>0;
      int activeRank=-1;

      if (iAmActive)
      {
	MPI_Comm_rank(commActive,&activeRank);
	// we assume points2octree distributes the tree on several first processors, without skipping through processors
	assert(activeRank==mpiRank());

	std::vector<ot::TreeNode> blocks;
	std::vector<ot::TreeNode> mins;
	blockPartStage1(linOct, blocks, 3/*dim*/, maxLevel(), commActive);
	blockPartStage2(linOct, blocks, mins, 3/*dim*/, maxLevel(), commActive);
	//blocks will be sorted.

	assert(!blocks.empty());
	assert(!linOct.empty());

	MPI_Comm_free(&commActive);
      }
    }
    else if (0==strcmp(repartition_type,"uniformOct"))
    {
      if (!mpiRank())
	cout<<"Repartitioning finest octree equally between processors"<<endl;
      par::partitionW<ot::TreeNode>(linOct, NULL, mpiComm());
    }
    else if (0==strcmp(repartition_type,"weighted_leaves"))
    {
      if (!mpiRank())
	cout<<"Repartitioning finest octree using heuristic weights for leaves"<<endl;
      // create "mins" array on each process
      // mins[i] is octant whos anchor is the start of the area owned by process [i]. Thus
      // mins[0] is supposed to have anchor at the origin
      // final size of mins will be number of "active" processes
      CreateMins(linOct);

      // -------- re-distribute source and target coordinates  ---------
      // since we only estimate and balance the load this time, we won't destroy original _srcPos and _trgPos
      Vec srcPos_backup=_srcPos;
      Vec trgPos_backup=_trgPos;

      bool sources_and_targets_coincide = _srcPos==_trgPos;
      Vec newSrcPos;
      RedistrPoints(_srcPos, newSrcPos,  newSrcGlobalIndices);
      _srcPos = newSrcPos;

      // redistribute targets; 
      // (if sources and targets coincide, then just update pointer to target positions)
      if (sources_and_targets_coincide)
	_trgPos = newSrcPos;
      else
      {
	Vec newTrgPos;
	RedistrPoints(_trgPos, newTrgPos,  newTrgGlobalIndices);
	_trgPos=newTrgPos;
      }

      // ------- create local multilevel tree based on local leaves ----------
      assert(_nodeVec.size()==0); 
      CreateAncestors(linOct, _nodeVec);

      pC( srcData(false /* don't output statistics */) );  
      pC( trgData() );  

      // calculate the weight for each leaf
      pC(calcLeafWeights(linOct));

      long long locMax=-1, globMax, locSum=0, globSum;
      for(size_t i=0; i<linOct.size(); i++)
      {
	long long tmp;
	tmp = linOct[i].getWeight();
	locSum += tmp;
	locMax = max(locMax, tmp);
      }

      MPI_Allreduce ( &locMax, &globMax, 1, MPI_LONG_LONG_INT, MPI_MAX, mpiComm() );
      MPI_Allreduce ( &locSum, &globSum, 1, MPI_LONG_LONG_INT, MPI_SUM, mpiComm() );
      if (!mpiRank())
	cout<<"Max leaf weight: "<<globMax<<" Total problem weight: "<<globSum<<endl;

      // restore original positions for sources and targets
      VecDestroy(_srcPos);
      _srcPos=srcPos_backup;
      if (!sources_and_targets_coincide)
	VecDestroy(_trgPos);
      _trgPos=trgPos_backup;

      //  -------- clean up the temporary tree -----------
      _nodeVec.clear();
      _ctb2GlbSrcNodeMap.clear();
      _ctb2GlbSrcExaMap.clear();
      _usr2GlbSrcNodeMap.clear();
      _usr2GlbSrcExaMap.clear();

      // repartiniton leaves according to weights computed above
      if (globMax <= globSum/mpiSize())
	par::partitionW<ot::TreeNode>(linOct,ot::getNodeWeight,mpiComm());
      else
	// partitionW interleaves empty and non-empty processors if maximum leaf weight is greater than average per-processor weight
	// so we'll just skip load balancing in this case
	if (mpiRank()==0)
	  cout<<"Maximum leaf weight is greater than average per-processor weight, skipping load-balancing"<<endl;

      // debugging code -- check if inactive processors are interleaved with active
      int iAmActive = !linOct.empty();
      int activeSize;
      MPI_Allreduce(&iAmActive,&activeSize,1,MPI_INT,MPI_SUM,mpiComm());
      if (!mpiRank())
	cout<<"ActiveSize: "<<activeSize<<endl;
      if (mpiRank()<activeSize && !iAmActive)
	cout<<"Process "<<mpiRank()<<" is inactive"<<endl;
    }

  // create "mins" array on each process
  // mins[i] is octant whos anchor is the start of the area owned by process [i]. Thus
  // mins[0] is supposed to have anchor at the origin
  // final size of mins will be number of "active" processes
  CreateMins(linOct);


  // -------- re-distribute source and target coordinates  ---------
  bool sources_and_targets_coincide = _srcPos==_trgPos;

  Vec newSrcPos;
  RedistrPoints(_srcPos, newSrcPos,  newSrcGlobalIndices);
  // not clear if we should leave the destruction of this vector to the user
  VecDestroy(_srcPos);
  _srcPos = newSrcPos;

  // redistribute targets; 
  // (if sources and targets coincide, then just update pointer to target positions)
  if (sources_and_targets_coincide)
    _trgPos = newSrcPos;
  else
  {
    Vec newTrgPos;
    RedistrPoints(_trgPos, newTrgPos,  newTrgGlobalIndices);
    // not clear if we should leave the destruction of this vector to the user
    VecDestroy(_trgPos);
    _trgPos=newTrgPos;
  }

#if 0
  // ------------- temporary fix: replicate global finest octree on all processors -------------
  // distribute sizes 
  int lTreeSize = linOct.size();
  vector<int> TreeSizes(mpiSize());
  MPI_Allgather ( &lTreeSize, 1/*sendcount*/, MPI_INT, &TreeSizes[0], 1/*recvcount*/, MPI_INT, mpiComm() );

 // calculate displacements and total tree  size, reserve memory
  vector<int> displs(mpiSize());
  displs[0]=0;
  for (int i=1; i<mpiSize(); i++)
    displs[i] = displs[i-1] + TreeSizes[i-1];
  
  int gTreeSize=displs.back()+TreeSizes.back(); // global tree size

  // reserve memory
  vector<ot::TreeNode> globalTree(gTreeSize);

  
  // do AllGatherv
  MPI_Allgatherv ( 
      linOct.size()? &linOct[0]:0,  //avoid crash with -D_GLIBCXX_DEBUG 
      lTreeSize,
      par::Mpi_datatype<ot::TreeNode >::value() , 
      &globalTree[0], 
      &TreeSizes[0], 
      &displs[0], 
      par::Mpi_datatype<ot::TreeNode >::value(), 
      mpiComm()
      );

  linOct.clear(); 
#endif 

  // ------- create local multilevel tree based on local leaves ----------
  assert(_nodeVec.size()==0); 
  CreateAncestors(linOct, _nodeVec);

  // print some statistics about finest-level tree
  unsigned globMaxLevel, globMinLevel;
  {
    unsigned maxLevel=0, minLevel=1000;

    for (size_t i=0; i< linOct.size(); i++)
    {
      maxLevel=max(maxLevel,linOct[i].getLevel());
      minLevel=min(minLevel,linOct[i].getLevel());
    }

    MPI_Reduce ( &maxLevel, &globMaxLevel, 1, MPI_UNSIGNED, MPI_MAX, 0, mpiComm() );
    MPI_Reduce ( &minLevel, &globMinLevel, 1, MPI_UNSIGNED, MPI_MIN, 0, mpiComm() );
  }

  if (!mpiRank())
    cout<<"Max level of octant in FINEST global tree: "<<globMaxLevel<<" Min level: "<<globMinLevel<<endl;

  PetscInt globalLeavesSize, minLocLeavesSize, maxLocLeavesSize;
  {
    PetscInt locLeavesSize = linOct.size();
    MPI_Reduce (&locLeavesSize, &globalLeavesSize, 1, MPIU_INT, MPI_SUM, 0, mpiComm() );
    MPI_Reduce (&locLeavesSize, &minLocLeavesSize, 1, MPIU_INT, MPI_MIN, 0, mpiComm() );
    MPI_Reduce (&locLeavesSize, &maxLocLeavesSize, 1, MPIU_INT, MPI_MAX, 0, mpiComm() );
  }

  if (!mpiRank())
  {
    cout<<"Finest octree sizes: global="<<globalLeavesSize<<" MaxLocal="<<maxLocLeavesSize<<" MaxLocal/MinLocal=";
    if (minLocLeavesSize==0)
      cout<<"inf";
    else
      cout<<double(maxLocLeavesSize)/minLocLeavesSize;
    cout<<endl;
  }

  linOct.clear(); //we no longer need our portion of leaves
  
  pC( srcData(true /* do print statist. */ ) );  
  pC( trgData() );  

  return(0);
}

// this function creates 2 VecScatter contexts necessary for re-distribution of source/target  locations, normals, charges, values of the potential
// it takes as input  PETSc vector "srcPos" of locations and array "procMins"  containing first blocks of active processors
// it returns STL vector of new global indices for each local source
// note that 1 index per source is returned, while each source is represented by 3 consecutive double numbers in input
#undef __FUNCT__
#define __FUNCT__ "CreateIndicesForRedistr"
void CreateIndicesForRedistr(Vec srcPos, const vector<ot::TreeNode> & procMins, MPI_Comm mpiComm, vector<PetscInt> & newScalarInd, PetscInt & newLocSize)
{
  PetscInt begLclPos, endLclPos;  
  VecGetOwnershipRange(srcPos, &begLclPos, &endLclPos); 
  begLclPos /= 3;
  endLclPos /= 3;
  PetscInt numLclPos = endLclPos-begLclPos;

  vector<int> partition(numLclPos);
  unsigned maxLevel = procMins[0].getMaxDepth();

  double* posArr; 
  VecGetArray(srcPos, &posArr);
  for(PetscInt k=0; k<numLclPos; k++) 
  {
    // convert current point to octant and then do binary search in procMins
    // we assume each coordinate is non-negative and STRICTLY LESS THAN 1.0
    unsigned x = static_cast<unsigned>(ldexp(posArr[3*k], maxLevel ));
    unsigned y = static_cast<unsigned>(ldexp(posArr[3*k+1], maxLevel ));
    unsigned z = static_cast<unsigned>(ldexp(posArr[3*k+2], maxLevel ));
    ot::TreeNode oct (x,y,z,maxLevel,3/*dim*/,maxLevel);

    vector<ot::TreeNode>::const_iterator it=upper_bound(procMins.begin(),procMins.end(),oct);

#ifdef DEBUG_LET
    assert(it>procMins.begin());
#endif
    it--; //we actually need the last element which is not greater than oct

    partition[k]=it-procMins.begin();  // particle "k" will go to processor it-procMins.begin()
  }
  VecRestoreArray(srcPos, &posArr);

  // following code borrowed from PETSc function ISPartitioningToNumbering
  int mpiSize;
  int mpiRank;
  MPI_Comm_size(mpiComm, &mpiSize);
  MPI_Comm_rank(mpiComm, &mpiRank);
  
  vector<PetscInt> lsizes(mpiSize,0);

  for (PetscInt i=0; i<numLclPos; i++) 
    lsizes[partition[i]]++;

  vector<PetscInt> sums(mpiSize);

  // MPIU_INT is defined inside petsc and corresponds to PetscInt
  MPI_Allreduce(&lsizes[0],&sums[0],mpiSize,MPIU_INT,MPI_SUM,mpiComm);
  
  // save the number of resulting elements on this process
  newLocSize=sums[mpiRank];
  
  vector<PetscInt> starts(mpiSize);
  MPI_Scan(&lsizes[0],&starts[0],mpiSize,MPIU_INT,MPI_SUM,mpiComm);
  
  for (int i=0; i<mpiSize; i++) 
    starts[i] -= lsizes[i];
  
  for (int i=1; i<mpiSize; i++) {
    sums[i]    += sums[i-1];
    starts[i]  += sums[i-1];
  }

  // For each local index give it the new global number
  newScalarInd.resize(numLclPos);
  for (PetscInt i=0; i<numLclPos; i++) 
    newScalarInd[i] = starts[partition[i]]++;
}


// this function builds all ancestors (up to root) of local leaves
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::CreateAncestors"
void Let3d_MPI::CreateAncestors(const vector<ot::TreeNode> & localLeaves, vector<Node> & _nodeVec)
{
  _nodeVec.reserve(localLeaves.size()+localLeaves.size()/7);  // estimate total number of nodes using infinite geometric progression: a+a/8+a/64+...
//  cout<<"rank "<<mpiRank()<<" reserved for nodevec: "<<localLeaves.size()+localLeaves.size()/7<<endl;
  
  /* Push the root node.  No parent, no children (so far),  path2Node = (0,0,0), depth 0 */
  _nodeVec.push_back( Node(-1,-1,Index3(0,0,0), 0) );
  
  if (localLeaves.size()==0 || (localLeaves.size()==1 && localLeaves[0]==ot::TreeNode (3/*dim*/, maxLevel()))) // degenerate case when we have only root octant in local tree
    ; // do nothing
  else
  {
    // first child of root will have index "1" :
    _nodeVec[0].chd()=1;
    // push 8 children of root
    for(int a=0; a<2; a++) {
      for(int b=0; b<2; b++) {
	for(int c=0; c<2; c++) {
	  /* Create a new node with parent at location "0"
	   * child initially set to -1 (changed when new node is looked at)
	   * path set to 2*"0's path" + the binary index of the new node's location relative to 0
	   * depth is set to "0's" depth + 1
	   */
	  _nodeVec.push_back( Node(0,-1, 2*_nodeVec[0].path2Node()+Index3(a,b,c), _nodeVec[0].depth()+1) );
	}
      }
    }

    // p is an index used to go around the tree; 
    // after processing of each leaf, p must contain index of its parent 
    // p initially contains index of root
    int p = 0;
    unsigned p_lev;  // level of p (0 corresponds to root)

    for(size_t n=0; n<localLeaves.size(); n++)
    {
      ot::TreeNode oct = localLeaves[n];
      unsigned x=oct.getX();
      unsigned y=oct.getY();
      unsigned z=oct.getZ();
      unsigned lev=oct.getLevel();

      //look for nearest common ancestor of last processed leaf (if any) and oct
      while(true) 
      {
#ifdef DEBUG_LET
	assert(p>=0);
#endif
	p_lev=_nodeVec[p].depth();

	// translate integer coordinates to DENDRO format
	unsigned tmpX = (_nodeVec[p].path2Node())(0)<<(maxLevel()-p_lev);
	unsigned tmpY = (_nodeVec[p].path2Node())(1)<<(maxLevel()-p_lev);
	unsigned tmpZ = (_nodeVec[p].path2Node())(2)<<(maxLevel()-p_lev);

	ot::TreeNode tmpOctant = ot::TreeNode(tmpX,tmpY,tmpZ,p_lev,3,maxLevel());
	if (tmpOctant.isAncestor(oct))
	  break;
	else
	  p=_nodeVec[p].par();
      }

      // now localLeaves[n] must be a descendant of some child of _nodeVec[p] (or child of _nodeVec[p] itself), and this child must be childless. This should hold since leaves in localLeaves are morton-sorted.

      while ( (p_lev=_nodeVec[p].depth()) < lev-1 )
      {
	// we need to add children to some child of p
	// first find out to which one
	Index3 idx;

#ifdef DEBUG_LET
	assert(lev>p_lev);
#endif
	idx(0)=(x>>(maxLevel()-p_lev-1)) & 1; 
	idx(1)=(y>>(maxLevel()-p_lev-1)) & 1; 
	idx(2)=(z>>(maxLevel()-p_lev-1)) & 1; 

	p = child(p, idx);  

	// debug
#ifdef DEBUG_LET
	assert(_nodeVec[p].chd()==-1);
#endif
	// now create children of p
	_nodeVec[p].chd()=_nodeVec.size();
	for(int a=0; a<2; a++) {
	  for(int b=0; b<2; b++) {
	    for(int c=0; c<2; c++) {
	      /* Create a new node with parent at location "p"
	       *  child initially set to -1 (changed when new node is looked at)
	       *  path set to 2*"p's path" + the binary index of the new node's location relative to p
	       *  depth is set to "k's" depth + 1
	       */
	      _nodeVec.push_back( Node(p,-1, 2*_nodeVec[p].path2Node()+Index3(a,b,c), _nodeVec[p].depth()+1) );
	    }
	  }
	}
      }

      // debug: check if integer coordinates of oct and appropriate child of p coincide  (up to format difference)
#ifdef DEBUG_LET
      Index3 idx;
      assert(lev==p_lev+1);
      idx(0)=(x>>(maxLevel()-p_lev-1)) & 1; 
      idx(1)=(y>>(maxLevel()-p_lev-1)) & 1; 
      idx(2)=(z>>(maxLevel()-p_lev-1)) & 1; 

      int q = child(p, idx);  
      assert(_nodeVec[q].chd()==-1);
      assert(oct.getX()==(unsigned)(_nodeVec[q].path2Node())(0)<<(maxLevel()-lev));
      assert(oct.getY()==(unsigned)(_nodeVec[q].path2Node())(1)<<(maxLevel()-lev));
      assert(oct.getZ()==(unsigned)(_nodeVec[q].path2Node())(2)<<(maxLevel()-lev));
#endif
    }
  }
}

namespace {
  // this structure type is used by function ExchangeOctantsNew
  struct octant_data
  {
    // we'll use the same format for coordinates as in "path2Node", NOT the same as in DENDRO; that is, coodinates of octants near root are small numbers, not padded with binary zeros from the right
    unsigned coord[3];
    unsigned level;

    int glbSrcExaNum;
    PetscInt glbSrcExaBeg;
    PetscInt glbSrcNodeIdx;
  };
}

/**
 * @author Ilya Lashuk
 * @brief This function implemements the exchange of octants between processes, so that for each octant the octant's data (see detailed description) is sent from the owner to all contributors and all users.
 *
 * Octant's data is: integer coordinates of octant and several indices: glbSrcExaBeg, glbSrcExaNum, glbSrcNodeIdx.
 * Flag LET_SRCENODE is also exchanged in the following sense: it is set for all octants which have owner.
 *
 * Algorithm for exchange is the following: owner of octant "o" sends "o" to all processes, whos areas intersect with the insulation layer of parent of "o".
 */
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::ExchangeOctantsNew"
void Let3d_MPI::ExchangeOctantsNew()
{
  // indToSend[i] will contain vector of indices of octants to be sent to process "i"
  vector<vector<size_t> > indToSend(procMins.size());

  if (_nodeVec[0].tag() & LET_OWNRNODE)  // if we own root octant...
  {
#ifdef DEBUG_LET
    assert(0);   // root octant should not be owned by anyone
#endif
    for (int cpu=0; (unsigned)cpu<procMins.size(); cpu++)
      if (cpu!=mpiRank())
	indToSend[cpu].push_back(0);  // ...then send root octant to every other active process
  }
  
  int q=0;  //  0 means root node;   
  while(q!=-1) 
  {
    // if node q is not a leaf
    // and the span of the insulation layer of this node is not entirely local
    // then  push all <children of q THAT WE OWN> to all processors whos areas intersect with the 
    // insulation layer
    // and go to first child of q and repeat
    if (!terminal(q))
    {
      unsigned q_lev=_nodeVec[q].depth();
      unsigned x,y,z;
      x = (_nodeVec[q].path2Node())(0) << maxLevel()-q_lev;
      y = (_nodeVec[q].path2Node())(1) << maxLevel()-q_lev;
      z = (_nodeVec[q].path2Node())(2) << maxLevel()-q_lev;

      ot::TreeNode oct (x,y,z,q_lev,3/*dim*/,maxLevel());
      ot::TreeNode root_oct(3/*dim*/, maxLevel());

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
      vector<ot::TreeNode>::const_iterator first_cpu = upper_bound(procMins.begin(),procMins.end(),min_neighb.getDFD());
      first_cpu--; //we actually need the last element which is not greater than search key

      // past_last_cpu is upper bound: any processor with rank>=past_last_cpu does not intersect with insul. layer 
      // we use getDLD() since max_neighb may intersect domains of many processors
      vector<ot::TreeNode>::const_iterator past_last_cpu = upper_bound(procMins.begin(),procMins.end(),max_neighb.getDLD());
      // now past_last_cpu points to first element which is greater than search key (maybe procMins.end() )

      if (past_last_cpu-first_cpu > 1)  // if insulation layer is not entirely local...
	// if insulation layer of an octant lies inside the area owned by single process, and this octant encloses some sources that our process owns, then this "single process" is OUR process
      {
	// domain_overlaps[k] will be set to 1 iff  domain of process "first_cpu-procMins.begin+k" indeed overlaps 
	// with insulation layer
	vector<char> domain_overlaps(past_last_cpu-first_cpu,0);
	vector<ot::TreeNode>::const_iterator my_cpu_it = procMins.begin()+mpiRank();
	for(size_t i=0; i<neighbs.size(); i++)
	{
	  vector<ot::TreeNode>::const_iterator nb_cpu_begin = upper_bound(first_cpu,past_last_cpu,neighbs[i].getDFD());
	  nb_cpu_begin--;
	  vector<ot::TreeNode>::const_iterator nb_cpu_end = upper_bound(first_cpu,past_last_cpu,neighbs[i].getDLD());

	  for (vector<ot::TreeNode>::const_iterator cpu_it=nb_cpu_begin; cpu_it<nb_cpu_end; cpu_it++)
	    if (cpu_it!=my_cpu_it)   // don't send anything to self
	      domain_overlaps[cpu_it-first_cpu]=1;
	}

	for (vector<ot::TreeNode>::const_iterator cpu_it=first_cpu; cpu_it<past_last_cpu; cpu_it++)
	  if (domain_overlaps[cpu_it-first_cpu])
	    // mark all children (which must exist, since node is non-terminal) THAT WE OWN
	    // to send to processor cpu_it
	    for (int chd_num=0; chd_num<8; chd_num++)
	    {
	      int chd_index=_nodeVec[q].chd()+chd_num;
	      if (_nodeVec[chd_index].tag() & LET_OWNRNODE)
		indToSend[cpu_it-procMins.begin()].push_back(chd_index);
	    }

	// go to first child of q 
	q=_nodeVec[q].chd();
	continue;  // (while loop)
      }
    } // end if (!terminal(q))

    // otherwise (i.e. if q is a leaf, or does not enclose any sources, or its insulation 
    // layer is  entirely local) go to "next" node 
    // next node is next sibling (if we are not last sibling or root), or next sibling of a parent 
    // (if parent is not last sibling itself) and so on; if we at the last node, just exit
    do
    {
      int p=_nodeVec[q].par();
      if (p==-1)
      {
	// q is root octant, thus there are no more nodes to process
	q=-1; // exit flag
	break;
      }
      if (q - _nodeVec[p].chd() < 7 )   // if q is not the last child ... (there are 8 children overall)
      {
	q++; // go to next sibling
	break;
      }
      else
	q=p; // go to parent and see what sibling parent is
    }
    while (true);

  } // end while(q!=-1)

  // now we need to send the octants we marked and receive octants from other processors

  // distribute sizes 
  // we'll use global communicator, just send/recv nothing to inactive processes
  // (processes with ranks >= mpiSize() )
  vector<int> sizesToSend(mpiSize(),0);
  for(size_t i=0; i<procMins.size(); i++)
    sizesToSend[i]=indToSend[i].size();
  vector<int> sizesToRecv(mpiSize(),0);

  PetscLogEventBegin(a2a_numOct_event,0,0,0,0);
  MPI_Alltoall( &sizesToSend[0], 1, MPI_INT, &sizesToRecv[0], 1, MPI_INT, mpiComm() );
  PetscLogEventEnd(a2a_numOct_event,0,0,0,0);

  assert(sizesToRecv[mpiRank()]==0);

  // register with MPI the structure to carry information about octants
  MPI_Datatype MPI_OCTDATA;
  MPI_Type_contiguous( sizeof(struct octant_data), MPI_BYTE, &MPI_OCTDATA);
  MPI_Type_commit(&MPI_OCTDATA);

 // calculate send and recv displacements and total size to send and receive, reserve memory
  vector<int> send_displs(mpiSize());
  vector<int> recv_displs(mpiSize());
  send_displs[0]=recv_displs[0]=0;
  for (int i=1; i<mpiSize(); i++)
  {
    send_displs[i] = send_displs[i-1] + sizesToSend[i-1];
    recv_displs[i] = recv_displs[i-1] + sizesToRecv[i-1];
  }
  
  int totalRecvSize=recv_displs.back()+sizesToRecv.back(); 
  int totalSendSize=send_displs.back()+sizesToSend.back(); 

  // reserve memory
  vector<struct octant_data> recvdOctants(totalRecvSize);
  vector<struct octant_data> octToSend_contig;
  octToSend_contig.reserve(totalSendSize);

  // pack data to send
  for (size_t cpu=0; cpu<procMins.size(); cpu++)
    for (size_t i=0; i<indToSend[cpu].size(); i++)
    {
      struct octant_data octData;
      Node & node = _nodeVec[indToSend[cpu][i]];
      
      Index3 path= node.path2Node();
      for (int j=0; j<3; j++)
	octData.coord[j]=path[j];

      octData.level=node.depth();

      octData.glbSrcExaBeg = node.glbSrcExaBeg();
      octData.glbSrcExaNum = node.glbSrcExaNum();
      octData.glbSrcNodeIdx = node.glbSrcNodeIdx();

      octToSend_contig.push_back(octData);
    }

  indToSend.clear();

  // print number of outgoing octants; the same octant sent to different processes is counted multiple times
  {
    PetscInt locNumOut = octToSend_contig.size();
    PetscInt maxNumOut, minNumOut;
    MPI_Reduce (&locNumOut, &maxNumOut, 1, MPIU_INT, MPI_MAX, 0, mpiComm() );
    MPI_Reduce (&locNumOut, &minNumOut, 1, MPIU_INT, MPI_MIN, 0, mpiComm() );
    if(!mpiRank())
      cout<<"ExchangeOctantsNew: number of outgoing octants, min="<<minNumOut<<" max="<<maxNumOut<<endl;
  }

  PetscLogEventBegin(a2aV_octData_event,0,0,0,0);
  // exchange octant data
  // when arr is empty,  &*arr.begin() and &arr[0] crash when using -D_GLIBCXX_DEBUG
  // that's why I use  "?:"
  MPI_Barrier(mpiComm());
  if (!mpiRank())
    cout<<"About to call MPI_Alltoallv to exchange ghost octant indices..."<<endl;
#ifdef USE_ALLTOALLV_FIX
  if (!mpiRank())
    cout<<"Using MPI_Alltoallv_viaSends for exchange of octant data"<<endl;
  MPI_Alltoallv_viaSends (
      octToSend_contig.size()==0? 0: &octToSend_contig[0], 
      &sizesToSend[0], 
      &send_displs[0], 
      MPI_OCTDATA, 
      recvdOctants.size()==0? 0: &recvdOctants[0],
      &sizesToRecv[0], 
      &recv_displs[0], 
      MPI_OCTDATA, 
      mpiComm());
#else
  MPI_Alltoallv (
      octToSend_contig.size()==0? 0: &octToSend_contig[0], 
      &sizesToSend[0], 
      &send_displs[0], 
      MPI_OCTDATA, 
      recvdOctants.size()==0? 0: &recvdOctants[0],
      &sizesToRecv[0], 
      &recv_displs[0], 
      MPI_OCTDATA, 
      mpiComm());
#endif
  PetscLogEventEnd(a2aV_octData_event,0,0,0,0);

  MPI_Barrier(mpiComm());
  if (!mpiRank())
    cout<<"Returned from MPI_Alltoallv"<<endl;
  
  MPI_Type_free(&MPI_OCTDATA);
  octToSend_contig.clear();
  sizesToSend.clear();
  send_displs.clear();
  sizesToRecv.clear();
  recv_displs.clear();

  // now insert received octants in local tree
  // some received octants may already be present in the tree, then just set the global indices
  // for now, we'll do simplistic implementation:  for each  received octant we start from root and go down the tree to find an appropriate place
  for (size_t i=0; i<recvdOctants.size(); i++)
  {
    int q= 0;  // we start from root octant
    const struct octant_data & octData = recvdOctants[i];

    while(true)
    {
#ifdef DEBUG_LET
      assert(unsigned(_nodeVec[q].depth()) <= octData.level);
#endif
      if (unsigned(_nodeVec[q].depth()) == octData.level)
      {
	// at this point "_nodeVec[q]" should be same octant as "octData"
	// thus set global indices
#ifdef DEBUG_LET
	Index3 path = _nodeVec[q].path2Node();
	for(int j=0; j<3; j++)
	  assert(unsigned(path[j])==octData.coord[j]);
#endif
	_nodeVec[q].glbSrcNodeIdx() = octData.glbSrcNodeIdx;
	_nodeVec[q].glbSrcExaNum() = octData.glbSrcExaNum;
	_nodeVec[q].glbSrcExaBeg() = octData.glbSrcExaBeg;
	_nodeVec[q].tag() |= LET_SRCENODE;
	break;
      }
      else // we must go deeper
      {
	// first, if _nodeVec[q] is leaf, create children of _nodeVec[q]
	if (_nodeVec[q].chd()==-1)
	{
	  _nodeVec[q].chd()=_nodeVec.size();
	  for(int a=0; a<2; a++) 
	    for(int b=0; b<2; b++)
	      for(int c=0; c<2; c++) 
	      {
		/* Create a new node with parent at location "q"
		 *  child initially set to -1 
		 *  path set to 2*"_nodeVec[q]'s path" + the binary index of the new node's location relative to _nodeVec[q]
		 *  depth is set to "k's" depth + 1
		 */
		_nodeVec.push_back( Node(q,-1, 2*_nodeVec[q].path2Node()+Index3(a,b,c), _nodeVec[q].depth()+1) );
	      }
	}
	
	// now go to the appropriate child of _nodeVec[q] (maybe just created by code above)
	unsigned idx[3];
	for(int j=0; j<3; j++)
	  idx[j]=( octData.coord[j] >> (octData.level - _nodeVec[q].depth() - 1) ) & 1; 
	q = _nodeVec[q].chd() + idx[0]*4+idx[1]*2+idx[2];
      }
    }
  }
}

/**
 * @author Ilya Lashuk
 * @brief This function implemements the exchange of octants between processes, so that in the end each process has in its local tree all ``foreign'' octants that it needs for U,V,W,X-lists.
 *
 * Node parameters glbSrcExaBeg, glbSrcExaNum, glbSrcNodeIdx and the flag LET_SRCENODE are also exchanged. In current temporary fix, LET_SRCENODE is set iff glbSrcNodeIdx is NOT -1.
 * 
 */
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::ExchangeOctants"
void Let3d_MPI::ExchangeOctants()
{
  // temporary fix to test correctness of new implementation of indices calculation
  // make values of glbSrcExaBeg, glbSrcExaNum and glbSrcNodeIdx for all nodes in global tree available to all processors
  // we assume here that each process has the full tree
  
  vector<PetscInt> exaBegS (_nodeVec.size());
  vector<PetscInt> finalExaBegS (_nodeVec.size());
  for (size_t i=0; i<_nodeVec.size(); i++)
    if (terminal(i) && (node(i).tag() & LET_OWNRNODE) )
      exaBegS[i]=_nodeVec[i].glbSrcExaBeg();
    else
      exaBegS[i]=-1;
  
  MPI_Allreduce( &(exaBegS[0]), &(finalExaBegS[0]), _nodeVec.size(), MPI_INT, MPI_MAX, mpiComm() ) ;
  exaBegS.clear();
  
  for (size_t i=0; i<_nodeVec.size(); i++)
    if (terminal(i) && (node(i).tag() & LET_OWNRNODE) )
      assert(_nodeVec[i].glbSrcExaBeg()==finalExaBegS[i]);
    else
      _nodeVec[i].glbSrcExaBeg()=finalExaBegS[i];
  finalExaBegS.clear();
  

  vector<int> exaNumS (_nodeVec.size());
  vector<int> finalExaNumS (_nodeVec.size());
  for (size_t i=0; i<_nodeVec.size(); i++)
    if (terminal(i) && (node(i).tag() & LET_OWNRNODE) )
      exaNumS[i]=_nodeVec[i].glbSrcExaNum();
    else
      exaNumS[i]=-1;
  
  MPI_Allreduce( &(exaNumS[0]), &(finalExaNumS[0]), _nodeVec.size(), MPI_INT, MPI_MAX, mpiComm() ) ;
  exaNumS.clear();
  
  for (size_t i=0; i<_nodeVec.size(); i++)
    if (terminal(i) && (node(i).tag() & LET_OWNRNODE) )
      assert(_nodeVec[i].glbSrcExaNum()==finalExaNumS[i]);
    else
      _nodeVec[i].glbSrcExaNum()=finalExaNumS[i];
  finalExaNumS.clear();


  vector<PetscInt> nodeIdxS (_nodeVec.size());
  vector<PetscInt> finalNodeIdxS (_nodeVec.size());
  for (size_t i=0; i<_nodeVec.size(); i++)
    if ( node(i).tag() & LET_OWNRNODE ) 
      nodeIdxS[i]=_nodeVec[i].glbSrcNodeIdx();
    else
      nodeIdxS[i]=-1;
  
  MPI_Allreduce( &(nodeIdxS[0]), &(finalNodeIdxS[0]), _nodeVec.size(), MPI_INT, MPI_MAX, mpiComm() ) ;
  nodeIdxS.clear();
  
  for (size_t i=0; i<_nodeVec.size(); i++)
    if ( node(i).tag() & LET_OWNRNODE ) 
      assert(finalNodeIdxS[i]==_nodeVec[i].glbSrcNodeIdx());
    else if (finalNodeIdxS[i]>=0)
    {
      _nodeVec[i].glbSrcNodeIdx()=finalNodeIdxS[i];
      // if node has global index other than -1, it means node is owned by some process which means we'll declare that the domain of the node encloses some sources
      node(i).tag() |= LET_SRCENODE;
    }
  finalNodeIdxS.clear();
}

#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::RedistrPoints"
/**
 * @author Ilya Lashuk
 * @brief This function implemements the redistribution of points according to the procMins array.
 *
 * @param oldPoints parallel PETSc vector with point coordinates (x1 y1 z1 x2 y2 z2 ... )
 * @param newPoints parallel PETSc vector for re-shuffled point coordinates. DO NOT call VecCreate() on this vector! VecCreateMPI() is called on this vector inside this function.
 * @param newIndices new global indices for local points of oldPoints are returned in this variable; For example, consider local portion of oldPoints; the first three numbers in this local portion define some point. newIndices[0] on exit will contain new global number of this point.
 */
void Let3d_MPI::RedistrPoints(Vec oldPoints, Vec & newPoints,  vector<PetscInt> & newIndices)
{
  // -------- re-distribute source coordinates  ---------
  PetscInt newLocSize;
  CreateIndicesForRedistr(oldPoints, procMins, mpiComm(), newIndices, newLocSize);

  // construct PETSc index set to redistribute source coordinates vector via VecScatter
  // basically, IS is a list of new global indices for local entries of oldPoints
  IS srcCoordIS;
  for (size_t i=0; i<newIndices.size(); i++)
    newIndices[i] *= 3;

  ISCreateBlock(
      mpiComm(),
      3, 
      newIndices.size(),
      newIndices.size()? &newIndices[0]:0,  //avoid crash with -D_GLIBCXX_DEBUG
      &srcCoordIS);

  // "newIndices" will be used later with possibly different block size, so divide it back
  for (size_t i=0; i<newIndices.size(); i++)
    newIndices[i] /= 3;

  // construct new vector and scatter context to redistribute source coordinates vector via VecScatter
  VecCreateMPI(mpiComm(),3*newLocSize,PETSC_DETERMINE/*global size*/,&newPoints);
  assert(procGlbNum(newPoints)==procGlbNum(oldPoints)); 

  VecScatter srcCoordScatterCtx;
  VecScatterCreate(oldPoints,PETSC_NULL,newPoints,srcCoordIS, &srcCoordScatterCtx);

  // do the actual communication 
  VecScatterBegin(srcCoordScatterCtx,oldPoints,newPoints,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(srcCoordScatterCtx,oldPoints,newPoints,INSERT_VALUES,SCATTER_FORWARD);


  VecScatterDestroy(srcCoordScatterCtx);
  ISDestroy(srcCoordIS);

  // ------- check if redistributed source coordinates indeed reside on correct processes ------
#ifdef DEBUG_LET
  {
    double* posArr; 
    VecGetArray(newPoints, &posArr);
    /* number of local positions for this processor */
    PetscInt numLclPos = procLclNum(newPoints);

    /* local range of indexes of particles */
    PetscInt begLclPos, endLclPos;  
    VecGetOwnershipRange(newPoints, &begLclPos, &endLclPos); 
    begLclPos /= 3;
    endLclPos /= 3;

    for(PetscInt k=0; k<numLclPos; k++) 
    {
      // convert current point to octant and then do binary search in procMins
      // we assume each coordinate is non-negative and STRICTLY LESS THAN 1.0
      unsigned x = (unsigned) ldexp(posArr[3*k], maxLevel() );
      unsigned y = (unsigned) ldexp(posArr[3*k+1], maxLevel() );
      unsigned z = (unsigned) ldexp(posArr[3*k+2], maxLevel() );
      ot::TreeNode oct (x,y,z,maxLevel(),3/*dim*/,maxLevel());

      vector<ot::TreeNode>::const_iterator it=upper_bound(procMins.begin(),procMins.end(),oct);

      assert(it-procMins.begin()==(mpiRank()+1));

    }
    VecRestoreArray(newPoints, &posArr);
  }
#endif
}

#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::srcData"
int Let3d_MPI::srcData(bool printStatistics)
{
  vector<Node>& nodeVec = this->nodeVec(); 

  /* Each processor puts its own srcdata into the tree  - new scope */
  /* local source number - number of source points in each box; different than above in different scope */
  vector<int> lclSrcNumVec;	 lclSrcNumVec.resize(nodeVec.size(), 0);
  /* Here vecIdxs[i] stores all indices for source positions
	* which are available to node i.  For example, vecIdxs[i].begin() to vecIdxs[i].end() store the indices of points
	* in node i or its descendants.  So, vecIdxs[0] stores the vector of indices of all points in the LET */
  vector< vector<PetscInt> > vecIdxs;	 vecIdxs.resize( nodeVec.size() );
  { 
	 /* get all source positions */
	 Vec pos = _srcPos;
	 double* posArr;	 pC( VecGetArray(pos, &posArr) );
	 /* get the number of positions for local porocessor */
	 // PetscInt numLclPos = procLclNum(pos);
	 /* get the range of indices that this processor is responsible for */
	 PetscInt begLclPos, endLclPos;  procLclRan(pos, begLclPos, endLclPos);

	 /* push all local indices from this processor's range into current indices list */
	 vector<PetscInt>& curVecIdxs = vecIdxs[0];
	 Point3 bbmin(ctr()-Point3(radius()));
	 Point3 bbmax(ctr()+Point3(radius()));  
	 for(PetscInt k=begLclPos; k<endLclPos; k++) {
		Point3 tmp(posArr+(k-begLclPos)*dim());
#ifdef DEBUG_LET
		pA(tmp>=bbmin && tmp<=bbmax);	 /* Assert this point is within the desired radius of the center */
#endif
		curVecIdxs.push_back(k);
	 }
	 /* lclSrcNumVec[0] stores number of all indices of points available to root of tree for this processor
	  * Specifically, the number of points this processor is in charge of storing in LET */
	 lclSrcNumVec[0] = curVecIdxs.size();

	 /* ordVec - ordering of the boxes in down->up fashion */
	 vector<int> ordVec;	 pC( dwnOrderCollect(ordVec) );
	 
	 for(size_t i=0; i<ordVec.size(); i++) {
		int curgNodeIdx = ordVec[i]; /* current node index */
		/* store all indices put into curVecIdxs into the current Node's vector of indices */
		vector<PetscInt>& curVecIdxs = vecIdxs[curgNodeIdx];

		/* If the current node is NOT a leaf (i.e., has children in the LET)
		 * then go through current vector of indices at build its childrens'
		 * vector of indices */
		if(terminal(curgNodeIdx)==false) {
		  Point3 curctr( center(curgNodeIdx) );
		  for(vector<PetscInt>::iterator pi=curVecIdxs.begin(); pi!=curVecIdxs.end(); pi++) {
			 Point3 tmp(posArr+(*pi-begLclPos)*dim());
			 Index3 idx;
			 for(int j=0; j<dim(); j++) {
				idx(j) = (tmp(j)>=curctr(j));
			 }
			 int chdgNodeIdx = child(curgNodeIdx, idx);
			 vector<PetscInt>& chdVecIdxs = vecIdxs[chdgNodeIdx];
			 chdVecIdxs.push_back(*pi);
		  }
		  curVecIdxs.clear(); //VERY IMPORTANT
		  /* For all of the current Node's children, put the vector of indices
			* into the local source number vector */
		  for(int a=0; a<2; a++) {
			 for(int b=0; b<2; b++) {
				for(int c=0; c<2; c++) {
				  int chdgNodeIdx = child(curgNodeIdx, Index3(a,b,c));
				  lclSrcNumVec[chdgNodeIdx] = vecIdxs[chdgNodeIdx].size();
				}
			 }
		  }
		} /* end if */
#ifdef DEBUG_LET_NONREGULAR
		else
		  assert(curVecIdxs.size()<=(size_t)ptsMax());
#endif
	 } /* end for loop through downward ordering of the nodes in the LET */
	 pC( VecRestoreArray(pos, &posArr) );
  } /* At the end of this scope, we've built lclSrcNumVec and VecIdxs for this processor */
  
  /* decide what nodes in LET we own */
  /* turn on the appropriate bits for nodes we own */
  // since eventually we'll only have ancestors of local leaves in LET at this point (plus some extra octants since every parent must have exactly 8 children), we won't bother to skip subtrees rooted at ``foreign'' octants (such subtrees contain root only)
  // we declare this process to be owner of an octant IFF:
  // a. anchor of the octant falls into area controlled by this processor
  // b. octant contains sources OR octant is non-leaf
  // That is, for nonleaves we don't check whether it encloses sources.
  // Leaves without sources won't be owned by any process and won't have global index and won't have LET_SRCENODE flag.
  // All other octants will have both global index and LET_SRCENODE flag, even if they don't enclose any sources. Nonleaf without
  // sources is rare situation, but this might happen, if, say, we balance finest octree.
  // c. Root octant won't be owned by anyone, and neither LET_CBTRNODE nor LET_OWNRNODE nor LET_SRCENODE will be set for root octant. Care is taken above to have at least one level of subdivision (at least 8 octants in global tree).

  _lclGlbSrcNodeCnt = 0; // _lclGlbSrcNodeCnt will be number of nodes owned by this process
  
  for(size_t i=1; i<_nodeVec.size(); i++)  // we start with i=1 to skip the root octant
    if (!terminal(i) || lclSrcNumVec[i]>0)
    {
      // convert octant's coordinates to dendro format
      unsigned lev = (unsigned)_nodeVec[i].depth();
      unsigned X = (unsigned)(_nodeVec[i].path2Node())(0)<<(maxLevel()-lev);
      unsigned Y = (unsigned)(_nodeVec[i].path2Node())(1)<<(maxLevel()-lev);
      unsigned Z = (unsigned)(_nodeVec[i].path2Node())(2)<<(maxLevel()-lev);

      ot::TreeNode oct (X,Y,Z,maxLevel(),3/*dim*/,maxLevel());
      vector<ot::TreeNode>::const_iterator it=upper_bound(procMins.begin(),procMins.end(),oct);
      it--; //we actually need the last element which is not greater than oct

      if ( it-procMins.begin() == mpiRank()  )
      {
	node(i).tag() |= LET_SRCENODE;  // we'll declare that this node encloses some sources,
	                                // although in rare cases it might not be true, see comment above
	node(i).tag() |= LET_OWNRNODE;
	_lclGlbSrcNodeCnt++;
      }
      

      if (lclSrcNumVec[i]>0)           // "if there are sources owned by this process and enclosed by this box" 
	node(i).tag() |= LET_CBTRNODE; // this process contributes to this node
                                       // note that this process might NOT own this node
    }
  
  /* based on the owner info, assign glbSrcNodeIdx, glbSrcExaBeg, glbSrcExaNum  */
  // also set global parameter _glbGlbSrcExaCnt
  // -------- assign glbSrcNodeIdx  and _glbGlbSrcNodeCnt  --------------	 
  PetscInt begNodeVec; //will store our first index for nodes (octants)
  MPI_Scan(&_lclGlbSrcNodeCnt,&begNodeVec,1,MPIU_INT,MPI_SUM,mpiComm());

  /* _glbGlbSrcNodeCnt is the total number of nodes which have owner, i.e. are "owned" by some processor; nodes without sources inside them (or inside their descendants) are not owned by any process */
  // broadcast from last process the total number of nodes which have owners (some nodes do not have owners)
  _glbGlbSrcNodeCnt = begNodeVec; // on last process begNodeVec is now total number of octants which have owners
  MPI_Bcast (&_glbGlbSrcNodeCnt, 1, MPIU_INT, mpiSize()-1, mpiComm() );

  if (printStatistics && !mpiRank())
    cout<<"Total number of octants which have owners: "<<_glbGlbSrcNodeCnt<<endl;

  begNodeVec -= _lclGlbSrcNodeCnt;
  
  // since sources are redistributed, the first "global" index of local sources is just begLclPos
  PetscInt begExaVec;
  VecGetOwnershipRange(_srcPos,&begExaVec,PETSC_NULL);
  begExaVec /= 3; /* 3 is dimension of a problem */
  
  // variables for statistics about non-empty leaves
  unsigned max_perOct=0, min_perOct=1u<<(sizeof(unsigned)*8-1);
  unsigned gl_max_perOct, gl_min_perOct;
  PetscInt num_active_leaves = 0;
  PetscInt gl_num_active_leaves;

  /* For each node, the following sets a way to index the nodes and sources using parameters set in each node
   * and available globally */
  for(size_t i=0; i<_nodeVec.size(); i++) 
  {
    // "poison memory"
    // among other things this guarantees that all octants which are non-leaves in global tree will have -1 in fields "glbSrcExaNum" and "glbSrcExaBeg" (after "ExchangeOctants")
    node(i).glbSrcNodeIdx() = -1;
    node(i).glbSrcExaNum()  = -1;
    node(i).glbSrcExaBeg()  = -1;

    if (node(i).tag() & LET_OWNRNODE)
    {
      node(i).glbSrcNodeIdx() = begNodeVec;
      begNodeVec ++;
      /* set "exact" source position variables for terminal/leaf nodes */
      if( terminal(i) ) {
	/* beginning index of source positions */
	node(i).glbSrcExaBeg() = begExaVec;
	begExaVec += lclSrcNumVec[i];

	/* exact number of sources */
	node(i).glbSrcExaNum() = lclSrcNumVec[i]; 

	/* save statistics */
	max_perOct=max(max_perOct,unsigned(lclSrcNumVec[i]));
	min_perOct=min(min_perOct,unsigned(lclSrcNumVec[i]));
	num_active_leaves++;
      }
    }
  }

  // print statistics about "active" leaf octants (the ones that have owners)
  MPI_Reduce(&max_perOct, &gl_max_perOct, 1, MPI_UNSIGNED, MPI_MAX, 0, mpiComm() );
  MPI_Reduce(&min_perOct, &gl_min_perOct, 1, MPI_UNSIGNED, MPI_MIN, 0, mpiComm() );
  MPI_Reduce(&num_active_leaves, &gl_num_active_leaves, 1, MPIU_INT, MPI_SUM, 0, mpiComm() );
  
  if(printStatistics && !mpiRank())
    cout<<gl_num_active_leaves<<" non-empty leaves globally, max pts per leaf: "<<gl_max_perOct<<"; min pts per non-empty leaf: "<<gl_min_perOct<<"; average pts per non-empty leaf: "<<double(procGlbNum(_srcPos))/gl_num_active_leaves<<endl;

  // exchange "ghost" octants; to be precise, owners of octants send out octant data (integer coordinates and several global indices) to potential contributors and potential users
  ExchangeOctantsNew();
  // ExchangeOctants();
  
  // print sizes of LETs -- maximum and min/max ratio

  PetscInt minLocLETSize, maxLocLETSize;
  {
    PetscInt locLETSize = _nodeVec.size();
    MPI_Reduce (&locLETSize, &minLocLETSize, 1, MPIU_INT, MPI_MIN, 0, mpiComm() );
    MPI_Reduce (&locLETSize, &maxLocLETSize, 1, MPIU_INT, MPI_MAX, 0, mpiComm() );
  }
  if (printStatistics && !mpiRank())
  {
    cout<<"LET sizes (with ghost octants): max="<<maxLocLETSize<<" max/min=";
    if (!minLocLETSize)
      cout<<"inf";
    else
      cout<< double(maxLocLETSize)/minLocLETSize;
    cout<<endl;
  }

  // now pad lclSrcNumVec with zeros, so it matches the size of _nodeVec
  // There should be no local sources enclosed by newly created octants.
  lclSrcNumVec.resize(_nodeVec.size(),0);

  /* global count of exact number of sources is just the size of _srcPos */
  _glbGlbSrcExaCnt = procGlbNum(_srcPos);
  /* local number of sources from the global count is the the number of sources for this processor */
  // looks like in this new implementation it will be just local number of sources (after re-distribution)
  _lclGlbSrcExaCnt = procLclNum(_srcPos);

  /* At end, will give total count of all nodes which have sources which contribute for this processor */
  _ctbSrcNodeCnt = 0;
  /* At end, will give total count of all source positions which contribute for this processor */
  _ctbSrcExaCnt = 0; 

  // reserve memory to avoid multiple re-allocations
  // in final version, almost every node in the LET will be "contributor", so pre-allocation size "_nodeVec.size()" should be reasonable
  _ctb2GlbSrcNodeMap.reserve(_nodeVec.size());
  _ctb2GlbSrcExaMap.reserve(procLclNum(_srcPos));

  for(size_t i=0; i<_nodeVec.size(); i++)
    if(node(i).tag() & LET_CBTRNODE)  // "if there are sources owned by this process and enclosed by this box" 
    {
#ifdef DEBUG_LET
      assert(lclSrcNumVec[i]>0); 
#endif
      /* Number of additional nodes (ancestors) thus far which contribute to this node with non-zero source numbers */
      node(i).ctbSrcNodeIdx() = _ctbSrcNodeCnt; 
      _ctbSrcNodeCnt ++;

      /* push map for contributor node to global source node index */
      _ctb2GlbSrcNodeMap.push_back( node(i).glbSrcNodeIdx() );

      /* For leaves set beginning index and exact number for contributor source positions */
      if(terminal(i)==true) { 
	node(i).ctbSrcExaBeg() = _ctbSrcExaCnt;
	node(i).ctbSrcExaNum() = lclSrcNumVec[i];
	/* Accumulate sum of the exact number of positions that contribute for this processor */
	_ctbSrcExaCnt += lclSrcNumVec[i];
	/* Set node(i)'s list of contributing source positions for this processor */
	node(i).ctbSrcOwnVecIdxs() = vecIdxs[i];
	/* Create a mapping from contributor source positions to global source position indices */
	for(size_t k=0; k < vecIdxs[i].size(); k++){
	  _ctb2GlbSrcExaMap.push_back( node(i).glbSrcExaBeg()+k );
	}
      }
    }

  return(0);
}

// ---------------------------------------------------------------------- 
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::trgData"
int Let3d_MPI::trgData()
{
  /* each proc put its own target data into the tree */
  vector<Node>& nodeVec = this->nodeVec();
  vector<int> lclSrcNumVec;  lclSrcNumVec.resize(nodeVec.size(), 0);
  vector< vector<PetscInt> > vecIdxs;  vecIdxs.resize(nodeVec.size());
  {
    Vec pos = _trgPos;
    double* posArr;	 pC( VecGetArray(pos, &posArr) );
    /* Get the number of positions for this processor */
    // PetscInt numLclPos = procLclNum(pos);
    /* Get the beginning and ending range values this processor is responsible for */
    PetscInt begLclPos, endLclPos;  procLclRan(pos, begLclPos, endLclPos);

    /* Create vector of indices of points for root of the LET */
    vector<PetscInt>& curVecIdxs = vecIdxs[0];
    Point3 bbmin(ctr()-Point3(radius()));
    Point3 bbmax(ctr()+Point3(radius())); 
    for(PetscInt k=begLclPos; k<endLclPos; k++) {
      Point3 tmp(posArr+(k-begLclPos)*dim());  
#ifdef DEBUG_LET
      pA(tmp>=bbmin && tmp<=bbmax);	 //LEXING: IMPORTANT : Asserts each point is within range of the center of the root
#endif
      curVecIdxs.push_back(k);
    }
    /* local number of sources for the root */
    lclSrcNumVec[0] = curVecIdxs.size();

    /* ordVec - ordering of the boxes in down->up fashion */
    vector<int> ordVec;	 pC( dwnOrderCollect(ordVec) );
    for(size_t i=0; i<ordVec.size(); i++) {
      int curgNodeIdx = ordVec[i]; /* current node index */
      vector<PetscInt>& curVecIdxs = vecIdxs[curgNodeIdx];

      /* Do the following for non-leaf nodes */
      if(terminal(curgNodeIdx)==false) {
	Point3 curctr( center(curgNodeIdx) );
	for(vector<PetscInt>::iterator pi=curVecIdxs.begin(); pi!=curVecIdxs.end(); pi++) {
	  /* Construct a point from the pointer as required for the current vector of indices
	   * Begins for root which is pre-built from above, and then all children indices of points will be built
	   * for use in as we traverse the ordVec list */
	  Point3 tmp(posArr+(*pi-begLclPos)*dim());
	  Index3 idx;
	  for(int j=0; j<dim(); j++) {
	    idx(j) = (tmp(j)>=curctr(j));
	  }
	  /* Retrieve the child of the current node based on its location relative to the point tmp
	   * in order to find where the point *pi resides (which child node) and then push that point
	   * into the child's vector of indices
	   */
	  int chdgNodeIdx = child(curgNodeIdx, idx);
	  vector<PetscInt>& chdVecIdxs = vecIdxs[chdgNodeIdx];
	  chdVecIdxs.push_back(*pi);
	} /* End of for loop for curVecIdxs */
	curVecIdxs.clear(); /* VERY IMPORTANT */
	/* For all 8 children of the current node, set the local number of sources based
	 * on the size of its vector of indices */
	for(int a=0; a<2; a++) {
	  for(int b=0; b<2; b++) {
	    for(int c=0; c<2; c++) {
	      int chdgNodeIdx = child(curgNodeIdx, Index3(a,b,c));
	      lclSrcNumVec[chdgNodeIdx] = vecIdxs[chdgNodeIdx].size();
	    }
	  }
	}
      }
    } /* End of for loop for ordVec traversal */
    pC( VecRestoreArray(pos, &posArr) );
  } /* End of this scope:  vecIdxs and lclSrcNumVec built */

  /* write evaTrgNodeIdx, evaTrgExaBeg, evaTrgExaNum
   * Here, for all nodes, we look to see if the local processor actually stores
   * any local sources there such that we know if the local processor is responsible
   * for evaluations at these nodes for these points */
  {
    _evaTrgNodeCnt = 0;
    _evaTrgExaCnt = 0;
    for(size_t i=0; i<nodeVec.size(); i++) {
      if(lclSrcNumVec[i]>0) {
	node(i).tag() |= LET_EVTRNODE; //LEXING - Turn on evaluator built
	node(i).evaTrgNodeIdx() = _evaTrgNodeCnt;
	_evaTrgNodeCnt ++;
	if(terminal(i)==true) {
	  node(i).evaTrgExaBeg() = _evaTrgExaCnt;
	  node(i).evaTrgExaNum() = lclSrcNumVec[i];
	  _evaTrgExaCnt += lclSrcNumVec[i];
	  node(i).evaTrgOwnVecIdxs() = vecIdxs[i];
	}
      }
    }
  }

  /* based on the owner info write usrSrcNodeIdx, usrSrcExaBeg, usrSrcExaNum
   * We will build the U,V,W, and X lists for each node in the LET, such that any box B
   * in these lists is used by the current processor.  That is, processor P is a "user" of B  */
  {
    for(size_t i=0; i<nodeVec.size(); i++){
      if(lclSrcNumVec[i]>0) {
	/* make sure that U,V,W, and X lists can be calculated/built properly.  See calGlbNodeLists(Node i) for more information */
	pC( calGlbNodeLists(i) );
	/* For all nodes in U,V,W, and X lists, turn on the bit to indicate that this processor
	 * uses these nodes for computation */
	for(vector<int>::iterator vi=node(i).Unodes().begin(); vi!=node(i).Unodes().end(); vi++)
	{
	  node(*vi).tag() |= LET_USERNODE;
#ifdef DEBUG_LET
	  assert(node(*vi).glbSrcExaBeg()>=0);
	  assert(node(*vi).glbSrcExaNum()>=0);
#endif
	}
	for(vector<int>::iterator vi=node(i).Vnodes().begin(); vi!=node(i).Vnodes().end(); vi++)
	  node(*vi).tag() |= LET_USERNODE;
	for(vector<int>::iterator vi=node(i).Wnodes().begin(); vi!=node(i).Wnodes().end(); vi++)
	  node(*vi).tag() |= LET_USERNODE;
	for(vector<int>::iterator vi=node(i).Xnodes().begin(); vi!=node(i).Xnodes().end(); vi++)
	{
	  node(*vi).tag() |= LET_USERNODE;
#ifdef DEBUG_LET
	  assert(node(*vi).glbSrcExaBeg()>=0);
	  assert(node(*vi).glbSrcExaNum()>=0);
#endif
	}
      }
    }

    /* Count the number of nodes being used by the current processor
     * and build a map between how these nodes are number and the global node indices.
     * Do the same for the source positions available to this processor
     */
    _usrSrcNodeCnt = 0;
    _usrSrcExaCnt = 0;
    for(size_t i=0; i<nodeVec.size(); i++) {
      if(node(i).tag() & LET_USERNODE) {
#ifdef DEBUG_LET
	assert(node(i).glbSrcNodeIdx()>=0);
#endif
	node(i).usrSrcNodeIdx() = _usrSrcNodeCnt;
	_usrSrcNodeCnt ++;
	_usr2GlbSrcNodeMap.push_back( node(i).glbSrcNodeIdx() );

	// terminal nodes in LET might be parent nodes in global tree;
	// thus, we instead need to check glbSrcExaNum or glbSrcExaBeg;
	// both are guaranteed to be -1 for parent nodes in global tree
	// and both guaranteed to be >=0 for leaves in global tree
	if(node(i).glbSrcExaNum()>=0)
	{
#ifdef DEBUG_LET
	  assert(node(i).glbSrcExaBeg()>=0);
#endif
	  node(i).usrSrcExaBeg() = _usrSrcExaCnt;
	  node(i).usrSrcExaNum() = node(i).glbSrcExaNum();
	  _usrSrcExaCnt += node(i).glbSrcExaNum();
	  for(PetscInt k=0; k<node(i).glbSrcExaNum(); k++){
	    _usr2GlbSrcExaMap.push_back( node(i).glbSrcExaBeg()+k );
	  }
	}
      }
    }
  } 

  return(0);
} /* end of trgData() */

// ----------------------------------------------------------------------
/* Build/Calculate the global node lists (U,V,W, and X lists) for a specific node */
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::calGlbNodeLists"
int Let3d_MPI::calGlbNodeLists(int gNodeIdx)
{
  //begin
  /* We use sets here since the values will be unnecessary to keep associated with the keys.
   * Can also use a C++ map, but it is unnecessary here to maintain unique key/value pairs
   * See let3d_mpi.hpp for descriptions of what U,B,W, and X lsists are. */
  set<int> Uset, Vset, Wset, Xset;  
  int curgNodeIdx = gNodeIdx;
  /* Make sure current node is not the root as the root has no ancestors */
  if(root(curgNodeIdx)==false) {
    /* get parent of node we're interested in */
    int pargNodeIdx = parent(curgNodeIdx);
    Index3 minIdx(0); /* = (0,0,0) */
    Index3 maxIdx(pow2(depth(curgNodeIdx))); /* = (2^d, 2^d, 2^d) */

    /* Try several different paths.  Here we are looking for paths which
     * will lead to children of neighbors of the parent of the current node
     * in order to build U,V,W, and X lists */
    for(int i=-2; i<4; i++) {
      for(int j=-2; j<4; j++) {
	for(int k=-2; k<4; k++) {
	  /* New Path to try */
	  Index3 tryPath2Node( 2*path2Node(pargNodeIdx) + Index3(i,j,k) );
	  /* Verify new path does not exceed a maximu index and is greater than (0,0,0) and is not equal to current node's path */
	  if(tryPath2Node >= minIdx && tryPath2Node <  maxIdx && tryPath2Node != path2Node(curgNodeIdx)) {
	    /* Look for the children of the neighbors of the current node's parent */
	    int resgNodeIdx = findGlbNode(depth(curgNodeIdx), tryPath2Node);
	    /* adj = true if nodes have at least one edge touching */
	    bool adj = adjacent(resgNodeIdx, curgNodeIdx);
	    /* When test node is higher in the tree than current node */
	    if( depth(resgNodeIdx)<depth(curgNodeIdx) ) {
	      /* If adjacent and current node is a leaf, put test node in the Uset if the current node is a leaf */
	      if(adj){
		if(terminal(curgNodeIdx)){ /* need to test if res a leaf?? HARPER */
#ifdef DEBUG_LET
		  assert(terminal(resgNodeIdx));
#endif
		  Uset.insert(resgNodeIdx);
		}
		else {;} /* non-leaves do not have U-lists */
	      }
	      else{
		/* The nodes are not adjacent, but resgNodeIdx is still a child of the current node's parent's neighbors.
		 * Hence, the current node's parent is adjacent to resgNodeIdx.  IN general. the current node is in the
		 * W-list of resgNodeIdx */
		Xset.insert(resgNodeIdx);
	      }
	    } /* End of "if depth(res) < depth(current) " */

	    /* Current node and test node at same depth */
	    if( depth(resgNodeIdx)==depth(curgNodeIdx) ) {
	      /* Two nodes at same depth not adjacent */
	      if(!adj) {
		Index3 bb(path2Node(resgNodeIdx)-path2Node(curgNodeIdx));
		/* Verify that no single component of the two paths exceed a limit */
#ifdef DEBUG_LET
		assert( bb.linfty()<=3 );
#endif
		/* resgNodeIdx is a child of the neighbor's of the currentr node's parents and not-adjacent to current node.
		 * Hence, it is in the V-list */
		Vset.insert(resgNodeIdx);
	      }
	      /* nodes at same depth and are adjacent, so resgNodeIdx could be in U OR W lists */
	      else {
		if(terminal(curgNodeIdx)) {
		  /* queue:  elements added/pushed to the back and removed/popped from the front */ 
		  queue<int> rest;
		  /* push resgNodeIdx into the queue */
		  rest.push(resgNodeIdx);
		  while(rest.empty()==false) {
		    int fntgNodeIdx = rest.front(); rest.pop(); /* Set front temp node index and pop the list */
		    /* If the current node and temp node are not adjacent, temp node is in W-list of current node */
		    if(adjacent(fntgNodeIdx, curgNodeIdx)==false) {
		      Wset.insert( fntgNodeIdx );
		    }
		    /* Current node and temp node ARE adjacent */
		    else {
		      /* If temp node is a leaf, it is in current node's U-list */
		      if(terminal(fntgNodeIdx)) {
			Uset.insert(fntgNodeIdx);
		      }
		      /* If temp node is not a leaf, then one of its descendants may be a leaf in the U or W lists of current node
		       * So, we push those into the queue
		       */
		      else { 
			for(int a=0; a<2; a++) {
			  for(int b=0; b<2; b++) {
			    for(int c=0; c<2; c++) {
			      rest.push( child(fntgNodeIdx, Index3(a,b,c)) );
			    }
			  }
			}
		      }
		    }
		  } /* End of while loop for rest queue */
		} /* End of "if current node a leaf" */ 
	      } /* End of res and current nodes at same depth and adjacent */
	    } /* End of "if depth(res) == depth(current)" */
	  } /* End of trypath */
	}
      }
    }
  } /* End of building sets for non-root node, curgNodeIdx */

  /* If the current node is a leaf, then it is in its own U-list */
  if(terminal(curgNodeIdx))
    Uset.insert(curgNodeIdx);

  /* For all sets, check to make sure all of the nodes actually have sources in them.  If not
   * we do not need to put them in the lists.  If so, build the U,V,W, and X lists from respective sets
   */
  for(set<int>::iterator si=Uset.begin(); si!=Uset.end(); si++)
    /* For nodes on U list check if they are indeed leaves with sources in global tree (e.g., glbSrcExaNum must be positive).
     * Note that checking LET_SRCENODE does not always work: LET_SRCENODE will be set for parent octant enclosing no sources,
     * this parent octant will be sent to us by other processor, but it's children might not be sent. Thus, the octant will 
     * be leaf in our local tree, and have LET_SRCENODE set. Also note that _glbSrcExaNum is initialized to -1 in the constructor of Node class, 
     * so it should be always safe to check _glbSrcExaNum */
    if(node(*si).glbSrcExaNum()>0) 
      node(gNodeIdx).Unodes().push_back(*si);
  for(set<int>::iterator si=Vset.begin(); si!=Vset.end(); si++)
    if(node(*si).tag() & LET_SRCENODE)
      node(gNodeIdx).Vnodes().push_back(*si);
  for(set<int>::iterator si=Wset.begin(); si!=Wset.end(); si++)
    if(node(*si).tag() & LET_SRCENODE)
      node(gNodeIdx).Wnodes().push_back(*si);
  for(set<int>::iterator si=Xset.begin(); si!=Xset.end(); si++)
    /* see comment for U-list */
    if(node(*si).glbSrcExaNum()>0)
      node(gNodeIdx).Xnodes().push_back(*si);

  return(0);
}
// ---------------------------------------------------------------------- 
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::dwnOrderCollect"
int Let3d_MPI::dwnOrderCollect(vector<int>& ordVec)
{
  /* ordVec - ordering of the boxes in downward fashion */
  ordVec.clear();
  for(size_t i=0; i<_nodeVec.size(); i++)
	 ordVec.push_back(i);
  return(0);
}
// ---------------------------------------------------------------------- 
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::upwOrderCollect"
int Let3d_MPI::upwOrderCollect(vector<int>& ordVec)
{
  /* ordVec - ordering of the boxes in upward fashion */
  ordVec.clear();
  for(int i=_nodeVec.size()-1; i>=0; i--)
	 ordVec.push_back(i);
  return(0);
}
// ----------------------------------------------------------------------
/* Return one of the eight children of a node based on a binary index, idx.  For example,
 * if idx = (0,0,0), the first child is returned.
 * If idx = (0,0,1), the second child is returned.
 * If idx = (0,1,0), the third child is returned.
 * If idx = (0,1,1), the fourth child is returned.
 * ...
 * If idx = (1,1,1), the eighth child is returned.
 */
int Let3d_MPI::child(int gNodeIdx, const Index3& idx)
{
#ifdef DEBUG_LET
  assert(idx>=Index3(0) && idx<Index3(2));
#endif
  return node(gNodeIdx).chd() + (idx(0)*4+idx(1)*2+idx(2));
}
/* Construct the center for a specific node */
Point3 Let3d_MPI::center(int gNodeIdx) 
{
  Point3 ll( ctr() - Point3(radius()) );
  int tmp = pow2(depth(gNodeIdx));
  Index3 path2Node_Lcl(path2Node(gNodeIdx));
  Point3 res;
  for(int d=0; d<dim(); d++) {
	 res(d) = ll(d) + (2*radius()) * (path2Node_Lcl(d)+0.5) / double(tmp);
  }
  return res;
}
/* Radius of a node =
 * radius(root)/(2^depth(node)).
 * When radius of root is 1 (most often),
 * radius of a node is 2^(-d)
 */
double Let3d_MPI::radius(int gNodeIdx) //radius of a node
{
  return radius()/double(pow2(depth(gNodeIdx)));
}
// ----------------------------------------------------------------------
/* Find a node based on a depth and some path.
 * This function is used to try to find children of the neighbors of the parent of some
 * node at depth wntDepth by trying several paths
 */
int Let3d_MPI::findGlbNode(int wntDepth, const Index3& wntPath2Node)
{
  int cur = 0;  
  Index3 tmpPath2Node(wntPath2Node);
  /* Keep trying to find a node cur which has depth greater than/equal to the given depth
	* OR is a terminal/leaf */
  while(depth(cur)<wntDepth && terminal(cur)==false) {
	 /* Difference in the depths */
	 int dif = wntDepth-depth(cur);
	 /* 2^(dif-1):  2^(difference in depths > 0) */
	 int tmp = pow2(dif-1);
	 /* Returns a binary index because of tmp's size */
	 Index3 choice( tmpPath2Node/tmp );

	 /* Get new path to follow */
	 tmpPath2Node -= choice*tmp;
	 /* Choose new node - hopefully this node is at the same depth as our current node */
	 cur = child(cur, choice);	 
  }  //cout<<endl;
  return cur;
}
// ----------------------------------------------------------------------
/* Returns true if two nodes have an edge or face which touch.  False otherwise */
bool Let3d_MPI::adjacent(int me, int you)
{
  int maxDepth = max(depth(me),depth(you)); /* Which node is at a greater depth */
  Index3 one(1); /* = (1,1,1) */
  /* Construct the center for both nodes */
  Index3 meCtr(  (2*path2Node(me)+one) * pow2(maxDepth - depth(me))  );
  Index3 youCtr(  (2*path2Node(you)+one) * pow2(maxDepth - depth(you))  );
  /* construct radii for both nodes */
  int meRad = pow2(maxDepth - depth(me));
  int youRad = pow2(maxDepth - depth(you));
  /* absolute value of difference of centers */
  Index3 dif( abs(meCtr-youCtr) );
  /* sum of radii */
  int radius  = meRad + youRad;
  /* if the abs. value of difference of centers is less than the sum of the radii
	* AND the infinity norm equals the sum of the radii, then the two nodes
	* are not too far away AND at least one edge touches */
  return
	 ( dif <= Index3(radius) ) && //not too far
	 ( dif.linfty() == radius ); //at least one edge touch
}

// ---------------------------------------------------------------------- 
#undef __FUNCT__
#define __FUNCT__ "Let3d_MPI::print"
int Let3d_MPI::print()
{
  //begin  //pC( PetscPrintf(MPI_COMM_WORLD, "nodeVec %d\n", _nodeVec.size()) );
  pC( PetscPrintf(MPI_COMM_SELF, "%d: %d glbGlbSrc %d %d  lclGlbSrc %d %d ctbSrc %d %d evaTrg %d %d usrSrc %d %d\n",
						mpiRank(), _nodeVec.size(),
						_glbGlbSrcNodeCnt, _glbGlbSrcExaCnt,   _lclGlbSrcNodeCnt, _lclGlbSrcExaCnt,
						_ctbSrcNodeCnt, _ctbSrcExaCnt,	 _evaTrgNodeCnt, _evaTrgExaCnt,			_usrSrcNodeCnt, _usrSrcExaCnt) );
  return(0);
}


