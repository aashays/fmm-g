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

/** \file
 * Header file for all FMM3d_MPI functions.  Most of the functions are located in
 * fmm3d_mpi.cpp
 * However, the evaluate function is in fmm3d_eval_mpi.cpp
 * The check function is in fmm3d_check_mpi.cpp
 * and the setup function is in fmm3d_setup_mpi.cpp
 * These functions are separated as they are the longest.
 */

#ifndef _FMM3D_HPP_
#define _FMM3D_HPP_

#include "knlmat3d_mpi.hpp"
#include "let3d_mpi.hpp"
#include "matmgnt3d_mpi.hpp"

//! FMM3d_MPI inherits from KnlMat3d_MPI.  Specific information for computing and using FMM are implemented here.  Specifically, setup and evaluate and inherits source positions, etc.  See knlmat3d_mpi.hpp for more information 
class FMM3d_MPI: public KnlMat3d_MPI
{
public:
  typedef pair<int,int> intpair;
  enum {	 /*! Upward Equivalent */ UE=0,
			 /*! Upward Check */ UC=1,
			 /*! Downward Equivalent */ DE=2,
			 /*! Downward Check */ DC=3,  };

  //------------------------------------
public:
  //! Extra node information.  Different from Let3d_MPI::NodeExt
  /*! NodeExt stores information about V-lists and effective data */
  class NodeExt {
  public:
	 /*! NodeExt constructor.  Initializes Vin and Vot variables to zero */
	 NodeExt(): _vLstInNum(0), _vLstInCnt(0), _vLstOthNum(0), _vLstOthCnt(0) {;}
	 /*! Number of boxes in V-list of this node */
	 int _vLstInNum;
	 /*! Count of points in V-list of this box */
	 int _vLstInCnt;
	 /*! Effective Values (data used when using FFTs) */
	 DblNumVec _effVal;
	 /*! Number of boxes which have this node in their V-lists */
	 int _vLstOthNum;
	 /*! Count of points affiliated with boxes which have this node in their V-lists */
	 int _vLstOthCnt;
	 /*! Effective densities (when using FFTs) */
	 DblNumVec _effDen;
};
  //! Node class for FMM3d_MPI stores access functions to NodeExt
  class Node {
  protected:
	 /*! Pointer to a NodeExt */
	 NodeExt* _ptr;
  public:
	 /*! Contrstruct Node by creating NULL pointer */
	 Node(): _ptr(NULL) {;}
	 /*! Desctroys FMM3d_MPI::NodeExt for this Node */
	 ~Node() {
		if(_ptr!=NULL)		  delete _ptr;
	 }
	 /*! If NodeExt does not exist here, create and return a pointer to it.  Otherwise, return a pointer to it */
	 NodeExt* ptr() {		if(_ptr==NULL)	_ptr = new NodeExt();		return _ptr;	 }
	 /*! Return NodeExt's _vLstInNum */
	 int& vLstInNum() { return ptr()->_vLstInNum; }
	 /*! Return NodeExt's _vLstInCnt */
	 int& vLstInCnt() { return ptr()->_vLstInCnt; }
	 /*! Return NodeExt's _effVal */
	 DblNumVec& effVal() { return ptr()->_effVal; }
	 /*! Return NodeExt's _vLstOthNum */
	 int& vLstOthNum() { return ptr()->_vLstOthNum; }
	 /*! Return NodeExt's _vLstOthCnt */
	 int& vLstOthCnt() { return ptr()->_vLstOthCnt; }
	 /*! Return NodeExt's _effDen */
	 DblNumVec& effDen() { return ptr()->_effDen; }
  };
  
  //------------------------------------
protected:
  /*! Center of FMM3d_MPI */
  Point3 _ctr;
  /*! the level of the root box, radius of the box is 2^(-_rootlvl) */
  int    _rootLevel; 
  /*! Level of precision */
  int _np;  
  //COMPONENTS, local member and data  vector<int> _stphds, _evlhds;
  /*! Local Essential Tree used for FMM3d_MPI */
  Let3d_MPI* _let;
  /*! 3D Matrix Managament for upward/downward check/equivalent information
	* It also provides multiplication between these contexts as well ae the maps between them
	* See matmgnt3d_mpi.hpp for more information */
  MatMgnt3d_MPI* _matmgnt;

  /*! Vector or nodes.  See Let3d_MPI for more information */
  vector<Node> _nodeVec;

  /*! Exact Source Positions - global vector */
  Vec _glbSrcExaPos;
  /*! Exact Source Normals - global vector */
  Vec _glbSrcExaNor;
  /*! Exact Source Densities - global vector */
  Vec _glbSrcExaDen;
  /*! Source upward equivalent densities- global vector */
  Vec _glbSrcUpwEquDen;

  /*! Contributor exact source positions vector */
  Vec _ctbSrcExaPos;
  /*! Contributor exact source normals vector */
  Vec _ctbSrcExaNor;
  /*! Contributor exact source densities vector */
  Vec _ctbSrcExaDen;
  /*! Contributor source upward equivalent densities vector */
  Vec _ctbSrcUpwEquDen;
  /*! Contributor exact source positions vector */
  Vec _ctbSrcUpwChkVal;
  /*! vecscatter manages communication of exact source positions between contributor and global contexts for parallel operation */
  VecScatter _ctb2GlbSrcExaPos;
  /*! vecscatter manages communication of exact source densities between contributor and global contexts for parallel operation */
  VecScatter _ctb2GlbSrcExaDen;
  /*! vecscatter manages communication of upward equivalent source densities between contributor and global contexts for parallel operation */
  VecScatter _ctb2GlbSrcUpwEquDen;

  /*! evaluator exact target positions vector */
  Vec _evaTrgExaPos;
  /*! evaluator exact target values vector */
  Vec _evaTrgExaVal;
  /*! evaluator target downward equivalent densities vector */
  Vec _evaTrgDwnEquDen;
  /*! evaluator target downward check values vector */
  Vec _evaTrgDwnChkVal;

  /*! user exact source positions vector */
  Vec _usrSrcExaPos;
  /*! user exact source normals vector */
  Vec _usrSrcExaNor;
  /*! user exact source densities vector */
  Vec _usrSrcExaDen;
  /*! user source upward equivalent densities vector */
  Vec _usrSrcUpwEquDen;
  /*! vecscatter manages communication of exact source positions between user and global contexts */
  VecScatter _usr2GlbSrcExaPos;
  /*! vecscatter manages communication of exact source densities between user and global contexts */
  VecScatter _usr2GlbSrcExaDen;
  /*! vecscatter manages communication of source upward equivalent densities between user and global contexts */
  VecScatter _usr2GlbSrcUpwEquDen;
  
  //IMPORTANT
  /*! This kernel allows for kernel-independence.  The type is established in the setup function in fmm3d_mpi_setup.cpp
	* Once the type is established, different souce and target positions can be used for up/down equivalent/check
	* multiplications for differrent mutipole/local->multipole/local operations in fmm3d_mpi_eval.cpp (where the
	* evaluate function is located */
  Kernel3d_MPI _knl_mm;
  
public:
  /*! Initialize FMM3d_MPI with string.  Calls constructor from KnlMat3d_MPI, and initializes are vectors here to NULL and values to zero.
	*  Center defaults to (0,0,0), rootLevel to 0, and np to 6 */
  FMM3d_MPI(const string& p);
  ~FMM3d_MPI();
  //MEMBER ACCESS
  /*! Return center (defaults to (0,0,0)) */
  Point3& ctr() { return _ctr; }
  /*! Return rootLevel (defaults to 0) */
  int& rootLevel() { return _rootLevel; }
  /*! Return precision level (defaults to 6 if not set in options file) */
  int& np() { return _np; }
  //SETUP and USE
  /*! setup function, detailed in fmm3d_mpi_setup.cpp */
  int setup();
  /*! evaluate function, detailed in fmm3d_mpi_eval.cpp */
  int evaluate(Vec srcDen, Vec trgVal);
  /*! check function, returns relative error.  Detailed in fmm3d_mpi_check.cpp */  
  int check(Vec srcDen, Vec trgVal, PetscInt numchk, double& rerr);  //int report();
  //OTHER ACCESS
  /*! Return local essential tree */
  Let3d_MPI* let() { return _let; }
  /*! Return matrix management */
  MatMgnt3d_MPI* matmgnt() { return _matmgnt; }

  /*! Return the node in nodeVec at the gNodeIdx position */
  Node& node(int gNodeIdx) { return _nodeVec[gNodeIdx]; }
  
protected:
  /*! Return the plain data size in _matmgnt, descibed by the enumerator type (UE, UC, DE, or DC) */
  int datSze(int tp) { return _matmgnt->plnDatSze(tp); }
  //multiplication
  /*! Source Equivalent to Target Check Multiplcation */
  int SrcEqu2TrgChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal);
  /*! Source Equivalent to Upward Check Multiplication */
  int SrcEqu2UpwChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, Point3 trgctr, double trgrad, const DblNumVec& srcDen, DblNumVec& trgVal);
  /*! Source Equivalent to Downward Check Mutiplcation */
  int SrcEqu2DwnChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, Point3 trgctr, double trgrad, const DblNumVec& srcDen, DblNumVec& trgVal);
  /*! Downward Equivalent to Target Check Mutiplcation */
  int DwnEqu2TrgChk_dgemv(Point3 srcctr, double srcrad, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal);
  /*! Upward Equivalent to Target Check Multiplcation */
  int UpwEqu2TrgChk_dgemv(Point3 srcctr, double srcrad, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal);
  //contributor data
  /*! Return matrix of contributor exact source positions for a specific node - gets information from _let */
  DblNumMat ctbSrcExaPos(int gNodeIdx);
  /*! Return matrix of contributor exact source normals for a specific node - gets information from _let */
  DblNumMat ctbSrcExaNor(int gNodeIdx);
  /*! Return vector of contributor exact source densities for a specific node - gets information from _let */
  DblNumVec ctbSrcExaDen(int gNodeIdx);
  /*! Return vector of contributor source upward equivalent densities for a specific node - gets information from _let */
  DblNumVec ctbSrcUpwEquDen(int gNodeIdx);
  /*! Return vector of contributor source upward check values for a specific node - gets information from _let */
  DblNumVec ctbSrcUpwChkVal(int gNodeIdx);
  
  //user data
  /*! Return matrix of user exact source positions for a specific node - gets information from _let */
  DblNumMat usrSrcExaPos(int gNodeIdx);
  /*! Return matrix of user exact source normals for a specific node - gets information from _let */
  DblNumMat usrSrcExaNor(int gNodeIdx);
  /*! Return vector of user source densities for a specific node - gets information from _let */
  DblNumVec usrSrcExaDen(int gNodeIdx);
  /*! Return vector of user source upward equivalent densities for a specific node - gets information from _let */
  DblNumVec usrSrcUpwEquDen(int gNodeIdx);
  
  //evaluator data
  /*! Return matrix of evaluator exact target positions for a specific node - gets information from _let */
  DblNumMat evaTrgExaPos(int gNodeIdx);
  /*! Return vector of evaluator exact target values for a specific node - gets information from _let */
  DblNumVec evaTrgExaVal(int gNodeIdx);
  /*! Return vector of evaluator target downward equivalent densities for a specific node - gets information from _let */
  DblNumVec evaTrgDwnEquDen(int gNodeIdx);
  /*! Return vector of evaluator target downward check values for a specific node - gets information from _let */
  DblNumVec evaTrgDwnChkVal(int gNodeIdx);
  
};
#endif
