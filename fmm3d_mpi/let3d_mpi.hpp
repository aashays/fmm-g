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
#ifndef _LET3D_HPP_
#define _LET3D_HPP_

#include "common/vec3t.hpp"
#include "common/nummat.hpp"
#include "comobject_mpi.hpp"
#include "TreeNode.h"

using std::vector;

enum {
  /*!this processor is a contributor to this (tree) node */
  LET_CBTRNODE = 1,
  /*!this processor is a user to this (tree) node */
  LET_USERNODE = 2,
  /*!this processor is a evaluator to this (tree) node */
  LET_EVTRNODE = 4,
  /*!this processor is the owner to this (tree) node */
  LET_OWNRNODE = 8,
  /*!this (tree) node has source points in it */
  LET_SRCENODE = 16 
};

//---------------------------------------------------------------------------
//! LET = Local Essential Tree.  See detailed description for more information.
/*! The LET is a global tree subset that a processor needs to evaluate the interaction on particles which belong to itself (owns).
 * That is, it is a unique identifier: a set of points and its bounding box.
 * To this extent, for a box/node  B, the information from B's lists are needed.
 * Specifically, the U, V, W and X lists are used for computation.
 * More information is in the descriptiptions of _Unodes, etc. and the papers.
 * For a processor P, P's LET contains the boxes which P "owns" and boxes in the U,V,W and X lists of these boxes.
 * For a box, B, owned by P, P is a contributor of B.  If B is in the U,V,W or X lists of a box P owns, P is a user of B.
 *  For more information, see "A New Parallel Kernel-Independent Fast Multipole Method" by Ying, Biros, Zorin and Langston.
 */
class Let3d_MPI: public ComObject_MPI
{
  //---------------------------------------
public:
  //! NodeExt stores information for a node such U, v, w, and X-lists as well as indices for contributor, evaluator and user nodes.
  class NodeExt {
  public:
	 /*! The Unodes or U-list of a box B are the leaf boxes adjacent to B, including itself.
	  * If B is a non-leaf, _Unodes is empty */
	 vector<int> _Unodes;
	 /*! The Vnodes or V-list of a box B contain the children of the neighbors of B's parent,
	  * which are not adjacent to B itself */
	 vector<int> _Vnodes;
	 /*! The Wnodes or W-list of a box B contain all boxes which are descendants of B's neighbors and are
	  * not adjacent to B but whose parents are adjacent to B.
	  * If B is not a leaf, _Wnodes is empty */
	 vector<int> _Wnodes;
	 /*! The Xnodes or X-list of a box B are all boxes, A, where B is in the W-list of A. */  
	 vector<int> _Xnodes;

	 /*! contributor source node index - built in srcData()*/
	 int _ctbSrcNodeIdx;
	 /*! contributor source exact beginning index */
	 int _ctbSrcExaBeg;
	 /*! contributor source exact number */
	 int _ctbSrcExaNum;
	 /*! contributor source node vector of indices - this is built in the function srcData() */
	 vector<PetscInt> _ctbSrcOwnVecIdxs;

	 /*! user source node index */
	 int _usrSrcNodeIdx;
	 /*! user source exact beginning index */
	 int _usrSrcExaBeg;
	 /*! user source exact number */
	 int _usrSrcExaNum;

	 /*! evaluator target node index */
	 int _evaTrgNodeIdx;
	 /*! evaluator target exact beginning */
	 int _evaTrgExaBeg;
	 /*! evaluator target exact number */
	 int _evaTrgExaNum;
	 /*! evaluator target vector of indices - built in trgData() */
	 vector<PetscInt> _evaTrgOwnVecIdxs;
  public:
	 /*! NodeExt() initilializes all indices to zero */
	 NodeExt(): 
		_ctbSrcNodeIdx(0), _ctbSrcExaBeg(0), _ctbSrcExaNum(0),
		_usrSrcNodeIdx(0), _usrSrcExaBeg(0), _usrSrcExaNum(0),
		_evaTrgNodeIdx(0), _evaTrgExaBeg(0), _evaTrgExaNum(0)
	 {;}
  };
  //! The Node class stores information for a node or box in the octree.  See for more detailed information.
  /*! Each box keeps a pointer to a NodeExt for extra information as well
	* as information on parents, children, depth, path to the node, etc.
	*/
  class Node {
  protected:
	 /*! Return integer location of parent node in node vector */
	 int _par;
	 /*! Return integer location of Node's first child in node vector */
	 int _chd;
	 /*! path to the node through the octree.  This 3 number code in binary represents how to traverse the tree to the node 
	  * That is, the integer vector (x,y,z) and its binary expansion tell how to go from the root box to the current box
	  * (0 stands for up/left and 1 for bottom/down, etc.).
	  */
	 Index3 _path2Node;
	 /*! tree-level depth of the node */ 
	 int _depth;
	 /*! global source node index - available to all processors*/
	 PetscInt _glbSrcNodeIdx;
	 /*! global source exact beginning - available to all processors*/
	 PetscInt _glbSrcExaBeg;
	 /*! global source exact number - available to all processors*/
	 int _glbSrcExaNum;
	 /*! specific for every processor - available to all processors*/
	 int _tag;
	 /*! Extra information for each node */
	 NodeExt* _ptr;

	 Node()
	 {
	   assert(false);
	 }
  public:
	 /*! Node is declared by providing integres which identify parent, first child, depth and path to this new node */
	 Node(int p, int c, Index3 t, int d):
		_par(p), _chd(c), _path2Node(t), _depth(d),
		_glbSrcNodeIdx(-1), _glbSrcExaBeg(-1),
		_glbSrcExaNum(-1), _tag(0), _ptr(NULL)  
         { }

	 
	 ~Node() {
		if(_ptr!=NULL)		  delete _ptr;
	 }
	 //access
	 /*! Return integer location of parent in node vector */
	 int& par() { return _par; }
	 /*! Return integer location of first child in node vector */
	 int& chd() { return _chd; }
	 /*! Return the path to the node */
	 Index3& path2Node() { return _path2Node; }
	 /*! Return node's depth in the tree */
	 int& depth() { return _depth; }
	 /*! Return node's tag */
	 int& tag() { return _tag; }
	 /*! Return the global source node index */
	 PetscInt& glbSrcNodeIdx() { return _glbSrcNodeIdx; }
	 /*! Return global source exact beginning index */
	 PetscInt& glbSrcExaBeg() { return _glbSrcExaBeg; }
	 /*! Return global exact number of sources */
	 int& glbSrcExaNum() { return _glbSrcExaNum; }

	 /*! Return pointer to NodeExt, which stores extra information */
	 NodeExt* ptr() {		if(_ptr==NULL) _ptr = new NodeExt();		return _ptr;	 }

	 /*! Return NodeExt's Unodes list */
	 vector<int>& Unodes() { return ptr()->_Unodes; }
	 /*! Return NodeExt's Vnodes list */	 
	 vector<int>& Vnodes() { return ptr()->_Vnodes; }
	 /*! Return NodeExt's Wnodes list */
	 vector<int>& Wnodes() { return ptr()->_Wnodes; }
	 /*! Return NodeExt's Xnodes list */
	 vector<int>& Xnodes() { return ptr()->_Xnodes; }

	 /*! Return a pointer to contributor source node index */
	 int& ctbSrcNodeIdx() { return ptr()->_ctbSrcNodeIdx; }
	 /*! Return a pointer to contributor source exact beginning index */
	 int& ctbSrcExaBeg() { return ptr()->_ctbSrcExaBeg; }
	 /*! Return a pointer to contributor source exact number */
	 int& ctbSrcExaNum() { return ptr()->_ctbSrcExaNum; }
	 /*! Return a pointer to contributor source vector of indices */
	 vector<PetscInt>& ctbSrcOwnVecIdxs() { return ptr()->_ctbSrcOwnVecIdxs; }

	 /*! Return a pointer to user source node index */
	 int& usrSrcNodeIdx() { return ptr()->_usrSrcNodeIdx; }
	 /*! Return a pointer to user source exact beginning index */
	 int& usrSrcExaBeg() { return ptr()->_usrSrcExaBeg; }
	 /*! Return a pointer to user source exact number */
	 int& usrSrcExaNum() { return ptr()->_usrSrcExaNum; }

	 /*! Return a pointer to evaluator target node index */
	 int& evaTrgNodeIdx() { return ptr()->_evaTrgNodeIdx; }
	 /*! Return a pointer to evaluator target exact beginning */
	 int& evaTrgExaBeg() { return ptr()->_evaTrgExaBeg; }
	 /*! Return a pointer to evaluator target exact number */
	 int& evaTrgExaNum() { return ptr()->_evaTrgExaNum; }
	 /*! Return a pointer to evaluator target vector of indices */
	 vector<PetscInt>& evaTrgOwnVecIdxs() { return ptr()->_evaTrgOwnVecIdxs; }
  };
  
  //---------------------------------------
protected:
  //PARAMS
  /*! source positions */
  Vec _srcPos;
  /*! target positions */
  Vec _trgPos;

  
  /*! Center */
  Point3 _ctr;
  /*! root level (Typically 0) */
  int _rootLevel;

  /*! Max number of points - not strictly enforced */
  int _ptsMax;
  /*! maximum number of levels */
  int _maxLevel;

  /*! vector of Nodes.  To return the ith Node, node(i) is called, returning _nodeVec[i] */
  vector<Node> _nodeVec;

  
  /*! global node count for global source vector - number of nodes (both leaves and non-leaves) WITH SOURCES in global tree; */
  PetscInt _glbGlbSrcNodeCnt;
  /*! global exact count for global source vector - exact number of so*/
  PetscInt _glbGlbSrcExaCnt;
  /*! local node count for global source vector - number of nodes in local essential tree of the global octree */
  PetscInt _lclGlbSrcNodeCnt;
  /*! local exact count for global source vector - information in local essential tree of global octree */
  PetscInt _lclGlbSrcExaCnt;

  /*! contributor source node count */
  int _ctbSrcNodeCnt;
  /*! contributor source exact count */
  int _ctbSrcExaCnt;
  /*! contributor-> global source node mapping */
  vector<PetscInt> _ctb2GlbSrcNodeMap;
  /*! contributor-> global source exact mapping */
  vector<PetscInt> _ctb2GlbSrcExaMap;

  /*! evaluator target node count */
  int _evaTrgNodeCnt;
  /*! evaluator target exact count */
  int _evaTrgExaCnt;

  /*! user source node count */
  int _usrSrcNodeCnt;
  /*! user source exact count */
  int _usrSrcExaCnt;
  /*! user -> global source node map */
  vector<PetscInt> _usr2GlbSrcNodeMap;
  /*! user -> global source exact map */
  vector<PetscInt> _usr2GlbSrcExaMap;
   
  // this function builds all ancestors (up to root) of local leaves
  void CreateAncestors(const vector<ot::TreeNode> & globalTree, vector<Node> & _nodeVec); 

  // void CreateIndicesForRedistr(Vec srcPos, const vector<ot::TreeNode> & procMins, MPI_Comm mpiComm, vector<PetscInt> & newScalarInd, PetscInt & newLocSize);

  void ExchangeOctants();
  
  // temporary, for debugging purposes only
  void ExchangeOctantsNew();
  // void ExchangeOctantsTreeBased();

  void RedistrPoints(Vec oldPoints, Vec & newPoints,  vector<PetscInt> & newIndices);
  void CreateMins(const vector<ot::TreeNode> & linOct);
  int  calcLeafWeights(vector<ot::TreeNode> & localLeaves);

public:
  /*! Construct Let3D_MPI with string for ComObject */
  Let3d_MPI(const string& p);
  ~Let3d_MPI();
  //MEMBER ACCESS
  /*! Return source positions */
  Vec& srcPos() { return _srcPos; }
  /*! Return target positions */
  Vec& trgPos() { return _trgPos; }
  
  /* Sources are redistributed during tree contruction; newSrcGlobalIndices[i] stores new global index for the source with old local index "i". Note that "i" is the index of the source, not of the particular coordinate. The coordinates have "old local" indices 3*i, 3*i+1, 3*i+2 and "new global" indices  3*newSrcGlobalIndices[i], 3*newSrcGlobalIndices[i]+1, 3*newSrcGlobalIndices[i]+2.
   */
  vector<PetscInt> newSrcGlobalIndices;
  
  /* targets are re-distributed during tree contruction, same as with sources. */
  vector<PetscInt> newTrgGlobalIndices;
  
  // first octants from all "active" processes; initialized in srcData()
  vector<ot::TreeNode> procMins;
  
  /*! Return center of tree */
  Point3& ctr() { return _ctr; }
  /*! Return root level (usually zero or one) */
  int& rootLevel() { return _rootLevel; }
  /*! return root's radius (2^(-rootLevel))*/
  double radius() { return pow(2.0, -_rootLevel); }
  /*! Return maximum number of points, set during setup from options file */
  int& ptsMax() { return _ptsMax; }
  /*! Return maximum number of levels in tree */
  int& maxLevel() { return _maxLevel; }
  //SETUP AND USE
  /*! setup function.  gets ptxMax and maxLevel as well as runs srcData() and trgData() for construction */
  int setup();
  /*! Print out information:  PetscPrintf(MPI_COMM_SELF, "%d: %d glbGlbSrc %d %d  lclGlbSrc %d %d ctbSrc %d %d evaTrg %d %d usrSrc %d %d\n",
	* mpiRank(), _nodeVec.size(),_glbGlbSrcNodeCnt, _glbGlbSrcExaCnt,   _lclGlbSrcNodeCnt, _lclGlbSrcExaCnt,
	*_ctbSrcNodeCnt, _ctbSrcExaCnt,	 _evaTrgNodeCnt, _evaTrgExaCnt,			_usrSrcNodeCnt, _usrSrcExaCnt) );
	*/
  int print();
  //construction
  /*! Assign all of the source data as necessary as well as set necessary variables for sources.
	* See let3d_mpi.cpp for more information */
  int srcData(bool);
  /*! Build target data variables.
	* See let3d_mpi.cpp for more information */
  int trgData();
  //---------------------------------------------
  //LOCAL
  /*! Build U,V,W, and X lists for a specific node, indicated by an integer location in the global node index. */
  int calGlbNodeLists(int gNodeIdx);
  /*!top down ordering of nodes */
  int dwnOrderCollect(vector<int>&);
  /*! bottom up ordering of nodes */
  int upwOrderCollect(vector<int>&); 

  /*! Return the vector of Nodes */
  vector<Node>& nodeVec() { return _nodeVec; }
  /*! Return a specific node in nodeVec using an integer locater.  I.e., _nodeVec[gNodeIdx] */
  Node& node(int gNodeIdx) { return _nodeVec[gNodeIdx]; }
  /*! If a node has no parent, then it is the root */
  bool    root(int gNodeIdx)     { return node(gNodeIdx).par()==-1; }
  /*! If a node has no children, then it is a leaf/terminal */
  bool    terminal(int gNodeIdx) { return node(gNodeIdx).chd()==-1; }
  /*! Return the integer locater (in nodeVec) of the parent of a node */
  int     parent(int gNodeIdx)   { assert(node(gNodeIdx).par()!=-1); return node(gNodeIdx).par(); }
  /*! Return child of node described by gNodeIdx and Idx.  gNodeIdx is the Node's integer descriptor in nodeVec
	* and Idx is a binary index from 0->7, indicating which child to return.
	* For example, child(10,(0,0,0)) returns the first child of nodeVec[10].
	* child(10,(1,1,1)) returns the eighth (and last) child of nodeVec[10].
	*/
  int     child( int gNodeIdx, const Index3& Idx);
  /*! Return an Index3 which gives a binary descriptor of the path to the node at nodeVec[gNodeIdx]
	* The integer vector (x,y,z) and its binary expansion tells how to go from the root box to the current box
	* (0 stands for up/left and 1 for bottom/down, etc.).
	*/
  Index3  path2Node(int gNodeIdx)     { return node(gNodeIdx).path2Node(); }
  /*! Return the depth in the octree of nodeVec[gNodeIdx] */
  int     depth(int gNodeIdx)    { return node(gNodeIdx).depth(); }
  /*! Return the center of the box dexcribed by nodeVec[gNodeIdx] */
  Point3  center(int gNodeIdx);
  /*! Return the radius of nodeVec[gNodeIdx]:
	* That is, the radius of the root divide by 2^(depth of the node at gNodeIx)
	* If the rootLevel is zero, then radius() is one, such that
	* radius(i) = 1/(2^(depth(i)))
	*/
  double  radius(int gNodeIdx);

  /*! Return global source node count, available to all processors */
  PetscInt& glbGlbSrcNodeCnt() { return _glbGlbSrcNodeCnt; }
  /*! Return global exact number of sources, available to all processors */
  PetscInt& glbGlbSrcExaCnt() { return _glbGlbSrcExaCnt; }
  /*! Return local source node count */
  PetscInt& lclGlbSrcNodeCnt() { return _lclGlbSrcNodeCnt; }
  /*! Return local exact number of sources */
  PetscInt& lclGlbSrcExaCnt() { return _lclGlbSrcExaCnt; }

  /*! Return number of contributor source nodes */
  int& ctbSrcNodeCnt() { return _ctbSrcNodeCnt; }
  /*! Return exact count of contributor sources */
  int& ctbSrcExaCnt() { return _ctbSrcExaCnt; }
  /*! Return vector of integers describing contributor to global source node map */
  vector<PetscInt>& ctb2GlbSrcNodeMap() { return _ctb2GlbSrcNodeMap; }
  /*! Return vector of integers describing contributor to global exact source map */
  vector<PetscInt>& ctb2GlbSrcExaMap() { return _ctb2GlbSrcExaMap; }

  /*! Return number of evaluator source nodes */
  int& evaTrgNodeCnt() { return _evaTrgNodeCnt; }
  /*! Return exact count of evaluator sources */
  int& evaTrgExaCnt() { return _evaTrgExaCnt; }

  /*! Return number of user source nodes */
  int& usrSrcNodeCnt() { return _usrSrcNodeCnt; }
  /*! Return exact count of user sources */
  int& usrSrcExaCnt() { return _usrSrcExaCnt; }
  /*! Return vector of integers describing user to global source node map */
  vector<PetscInt>& usr2GlbSrcNodeMap() { return _usr2GlbSrcNodeMap; }
  /*! Return vector of integers describing user to global exact source map */
  vector<PetscInt>& usr2GlbSrcExaMap() { return _usr2GlbSrcExaMap; }

  /*! Find a node in the global list based on a depth and path
	* This is used for trying to find the children of the neighbors
	* of the parent of some node at level "depth" for building
	* that node's U,V,W, and X lists
	*/
  int findGlbNode(int depth, const Index3& path2Node);
  /*! Check if nodeVec[me] is adjacent to nodeVec[you] */
  bool adjacent(int me , int you);

  /*! Return dimension.  Always returns 3 */
  int dim() const { return 3; }
  
  /*! For a certain processor, return local number of positions from Vec pos 
	* See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecGetLocalSize.html for more info */
  PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/dim(); }
  /*! For a certain processor, return global number of positions from Vec pos */
  PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/dim(); }
  /*! For a certain preocessor, return local range in beg and end for Vec pos of the local ownership
	* See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecGetOwnershipRange.html for more information */
  void procLclRan(Vec pos, PetscInt& beg, PetscInt& end) { VecGetOwnershipRange(pos, &beg, &end); beg=beg/dim(); end=end/dim(); }
};

#endif
