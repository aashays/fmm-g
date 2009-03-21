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
 * The file fmm3d_mpi.cpp provides implementations for all basic FMM3d functions except for the evaluate, setup, and check functions which have their own files.
 * Some multiplication and vriable access is taken care of here.
 */

#include "common/vecmatop.hpp"
#include "fmm3d_mpi.hpp"

/* Initialize all variables necessary for FMM3d, mostlt to NULL.
 * All of these will be built as necessary during setup.  See fmm3d_setup_mpi.cpp
 * for more information on where these get built.
 */
FMM3d_MPI::FMM3d_MPI(const string& p):
  KnlMat3d_MPI(p), _ctr(0,0,0), _rootLevel(0),
  _np(6),
  _let(NULL), _matmgnt(NULL),
  _glbSrcExaPos(NULL), _glbSrcExaNor(NULL), _glbSrcExaDen(NULL), _glbSrcUpwEquDen(NULL),
  _ctbSrcExaPos(NULL), _ctbSrcExaNor(NULL), _ctbSrcExaDen(NULL), _ctbSrcUpwEquDen(NULL),
  _ctbSrcUpwChkVal(NULL), _ctb2GlbSrcExaPos(NULL), _ctb2GlbSrcExaDen(NULL), _ctb2GlbSrcUpwEquDen(NULL),
  _evaTrgExaPos(NULL), _evaTrgExaVal(NULL), _evaTrgDwnEquDen(NULL), _evaTrgDwnChkVal(NULL),
  _usrSrcExaPos(NULL), _usrSrcExaNor(NULL), _usrSrcExaDen(NULL), _usrSrcUpwEquDen(NULL),
  _usr2GlbSrcExaPos(NULL), _usr2GlbSrcExaDen(NULL), _usr2GlbSrcUpwEquDen(NULL)
{
}

/* Destroy everything */
FMM3d_MPI::~FMM3d_MPI()
{
  if(_let!=NULL)	 delete _let;
  
  if(_glbSrcExaPos!=NULL) { iC( VecDestroy(_glbSrcExaPos) ); _glbSrcExaPos=NULL; }
  if(_glbSrcExaNor!=NULL) { iC( VecDestroy(_glbSrcExaNor) ); _glbSrcExaPos=NULL; }
  if(_glbSrcExaDen!=NULL) { iC( VecDestroy(_glbSrcExaDen) ); _glbSrcExaDen=NULL; }
  if(_glbSrcUpwEquDen!=NULL) { iC( VecDestroy(_glbSrcUpwEquDen) ); _glbSrcUpwEquDen=NULL; }
  
  if(_ctbSrcExaPos!=NULL) { iC( VecDestroy(_ctbSrcExaPos) ); _ctbSrcExaPos=NULL; }
  if(_ctbSrcExaNor!=NULL) { iC( VecDestroy(_ctbSrcExaNor) ); _ctbSrcExaNor=NULL; }
  if(_ctbSrcExaDen!=NULL) { iC( VecDestroy(_ctbSrcExaDen) ); _ctbSrcExaDen=NULL; }
  if(_ctbSrcUpwEquDen!=NULL) { iC( VecDestroy(_ctbSrcUpwEquDen) ); _ctbSrcUpwEquDen=NULL; }
  if(_ctbSrcUpwChkVal!=NULL) { iC( VecDestroy(_ctbSrcUpwChkVal) ); _ctbSrcUpwChkVal=NULL; }
  if(_ctb2GlbSrcExaPos!=NULL) { iC( VecScatterDestroy(_ctb2GlbSrcExaPos) ); _ctb2GlbSrcExaPos=NULL; }
  if(_ctb2GlbSrcExaDen!=NULL) { iC( VecScatterDestroy(_ctb2GlbSrcExaDen) ); _ctb2GlbSrcExaDen=NULL; }
  if(_ctb2GlbSrcUpwEquDen!=NULL) { iC( VecScatterDestroy(_ctb2GlbSrcUpwEquDen) ); _ctb2GlbSrcUpwEquDen=NULL; }
  
  if(_evaTrgExaPos!=NULL) { iC( VecDestroy(_evaTrgExaPos) ); _evaTrgExaPos=NULL; }
  if(_evaTrgExaVal!=NULL) { iC( VecDestroy(_evaTrgExaVal) ); _evaTrgExaVal=NULL; }
  if(_evaTrgDwnEquDen!=NULL) { iC( VecDestroy(_evaTrgDwnEquDen) ); _evaTrgDwnEquDen=NULL; }
  if(_evaTrgDwnChkVal!=NULL) { iC( VecDestroy(_evaTrgDwnChkVal) ); _evaTrgDwnChkVal=NULL; }
  
  if(_usrSrcExaPos!=NULL) { iC( VecDestroy(_usrSrcExaPos) ); _usrSrcExaPos=NULL; }
  if(_usrSrcExaNor!=NULL) { iC( VecDestroy(_usrSrcExaNor) ); _usrSrcExaNor=NULL; }
  if(_usrSrcExaDen!=NULL) { iC( VecDestroy(_usrSrcExaDen) ); _usrSrcExaDen=NULL; }
  if(_usrSrcUpwEquDen!=NULL) { iC( VecDestroy(_usrSrcUpwEquDen) ); _usrSrcUpwEquDen=NULL; }
  if(_usr2GlbSrcExaPos!=NULL) { iC( VecScatterDestroy(_usr2GlbSrcExaPos) ); _usr2GlbSrcExaPos=NULL; }
  if(_usr2GlbSrcExaDen!=NULL) { iC( VecScatterDestroy(_usr2GlbSrcExaDen) ); _usr2GlbSrcExaDen=NULL; }
  if(_usr2GlbSrcUpwEquDen!=NULL) { iC( VecScatterDestroy(_usr2GlbSrcUpwEquDen) ); _usr2GlbSrcUpwEquDen=NULL; }
}

// ----------------------------------------------------------------------
/* This function computes the source equivalent to target check multiplication */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::SrcEqu2TrgChk_dgemv"
int FMM3d_MPI::SrcEqu2TrgChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal)
{
  int TMAX = 1024;
  /* If the number of target positions is small, do one run */
  if(trgPos.n()<=TMAX) {
	 int M = trgPos.n() * _knl.trgDOF();
	 int N = srcPos.n() * _knl.srcDOF();
	 DblNumMat tmp(M,N);
	 /* Compute context in tmp for computation */
	 pC( _knl.buildKnlIntCtx(srcPos, srcNor, trgPos, tmp) );
	 /* Compute 1.0*tmp*srcDen + 1.0*trgVal */
	 pC( dgemv(1.0, tmp, srcDen, 1.0, trgVal) );
  }
  /* If the number of target positions is large, split up the multiplications over several smaller steps */
  else {
	 /* Number of runs */
	 int RUNS = (trgPos.n()-1) / TMAX + 1;
	 for(int r=0; r<RUNS; r++) {
		/* Where to start */
		int stt = r*TMAX;
		/* Where to end */
		int end = min((r+1)*TMAX, trgPos.n());
		/* Number of elements in the multiplcation process */
		int num = end-stt;
		int M = num * _knl.trgDOF();
		int N = srcPos.n() * _knl.srcDOF();
		/* Matrix of target positions */
		DblNumMat tps(dim(), num, false, trgPos.data() + stt*dim() );
		/* Vewctor of target values */
		DblNumVec tvl(num*_knl.trgDOF(), false, trgVal.data() + stt*_knl.trgDOF());
		DblNumMat tmp(M,N);
		/* Build the Kernel matrix and multiply for result */
		pC( _knl.buildKnlIntCtx(srcPos, srcNor, tps, tmp) );
		/* Compute 1.0*tmp*srcDen + 1.0*tvl */
		pC( dgemv(1.0, tmp, srcDen, 1.0, tvl) );
	 }
  }
  return(0);
}
// ----------------------------------------------------------------------
/* Compute Source Equivalent to Upward Check Multiplication */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::SrcEqu2UpwChk_dgemv"
int FMM3d_MPI::SrcEqu2UpwChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, Point3 trgCtr, double trgRad, const DblNumVec& srcDen, DblNumVec& trgVal)
{
  /* Build the target sample positions based on the upward check sample positions
	* as well as the target node's center and radius.  The resulting trgPos target
	* positions will be the UC sample positions for the requested node */
  DblNumMat trgPos; pC( _matmgnt->locPos(UC, trgCtr, trgRad, trgPos) );
  int M = trgPos.n() * _knl.trgDOF();
  int N = srcPos.n() * _knl.srcDOF();
  DblNumMat tmp(M,N);
  pC( _knl.buildKnlIntCtx(srcPos, srcNor, trgPos, tmp) );
  /* Compute 1.0*tmp*srcDen + 1.0*trgVal */
  pC( dgemv(1.0, tmp, srcDen, 1.0, trgVal) );
  return(0);
}
// ----------------------------------------------------------------------
/* Compute Equivalent Source to Downward Check Multiplication */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::SrcEqu2DwnChk_dgemv"
int FMM3d_MPI::SrcEqu2DwnChk_dgemv(const DblNumMat& srcPos, const DblNumMat& srcNor, Point3 trgCtr, double trgRad, const DblNumVec& srcDen, DblNumVec& trgVal)
{
  /* Build the target positions based on the downward check sample positions
	* and the node with center trgCtr and radius trgRad */
  DblNumMat trgPos; pC( _matmgnt->locPos(DC, trgCtr, trgRad, trgPos) );
  int M = trgPos.n() * _knl.trgDOF();
  int N = srcPos.n() * _knl.srcDOF();
  DblNumMat tmp(M,N);
  pC( _knl.buildKnlIntCtx(srcPos, srcNor, trgPos, tmp) );
  /* Compute 1.0*tmp*srcDen + 1.0*trgVal */
  pC( dgemv(1.0, tmp, srcDen, 1.0, trgVal) );
  return(0);
}
// ----------------------------------------------------------------------
/* Compute Downward Equivalent to Target Check Multiplication */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::DwnEqu2TrgChk_dgemv"
int FMM3d_MPI::DwnEqu2TrgChk_dgemv(Point3 srcCtr, double srcRad, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal)
{
  /* Target Max = 1024 */
  int TMAX = 1024;
  /* If the number of positions in trgPos is small, do one run.  Otherwise, we will break up the multiplication */
  if(trgPos.n()<=TMAX) {
	 /* Build sample source positions at the Downward Equivalent location
	  * based on the source node's center srcCtr and radius srcRad */
	 DblNumMat srcPos; pC( _matmgnt->locPos(DE, srcCtr, srcRad, srcPos) );
	 int M = trgPos.n() * _knl_mm.trgDOF();
	 int N = srcPos.n() * _knl_mm.srcDOF();
	 DblNumMat tmp(M,N);
	 /* Build kernel multiplier context and store in tmp */
	 pC( _knl_mm.buildKnlIntCtx(srcPos, srcPos, trgPos, tmp) );
	 /* Compute 1.0*tmp*srcDen + 1.0*trgVal */
	 pC( dgemv(1.0, tmp, srcDen, 1.0, trgVal) );
  }
  /* Break up multiplication into smaller steps */
  else {
	 /* Build sample source positions at the Downward Equivalent location
	  * based on the source node's center srcCtr and radius srcRad */
	 DblNumMat srcPos; pC( _matmgnt->locPos(DE, srcCtr, srcRad, srcPos) );
	 /* Decide how many runs we will break the multiplication into */
	 int RUNS = (trgPos.n()-1) / TMAX + 1;
	 for(int r=0; r<RUNS; r++) {
		/* Find start and end locations as well as number of elements being used */
		int stt = r*TMAX;
		int end = min((r+1)*TMAX, trgPos.n());
		int num = end-stt;
		int M = num * _knl_mm.trgDOF();
		int N = srcPos.n() * _knl_mm.srcDOF();
		/* Target positions matrix */
		DblNumMat tps(dim(), num, false, trgPos.data() + stt*dim());
		/* Target values vector */
		DblNumVec tvl(num*_knl_mm.trgDOF(), false, trgVal.data() + stt*_knl_mm.trgDOF());
		DblNumMat tmp(M, N);
		/* Build multiplier context */
		pC( _knl_mm.buildKnlIntCtx(srcPos, srcPos, tps, tmp) );
		/* Compute 1.0*tmp*srcDen + 1.0*tvl */
		pC( dgemv(1.0, tmp, srcDen, 1.0, tvl) );
	 }
  }
  return(0);
}

// ----------------------------------------------------------------------
/* Compute Upward Equivalent to Target Check Multiplication */
#undef __FUNCT__
#define __FUNCT__ "FMM3d_MPI::UpwEqu2TrgChk_dgemv"
int FMM3d_MPI::UpwEqu2TrgChk_dgemv(Point3 srcCtr, double srcRad, const DblNumMat& trgPos, const DblNumVec& srcDen, DblNumVec& trgVal)
{
  /* Target Max = 1024 */
  int TMAX = 1024;
  /* If the number of positions in trgPos is small, do one run.  Otherwise, we will break up the multiplication */
  if(trgPos.n()<=TMAX) {
	 /* Build sample source positions at the Upward Equivalent location
	  * based on the source node's center srcCtr and radius srcRad */
	 DblNumMat srcPos; pC( _matmgnt->locPos(UE, srcCtr, srcRad, srcPos) );
	 int M = trgPos.n() * _knl_mm.trgDOF();
	 int N = srcPos.n() * _knl_mm.srcDOF();
	 DblNumMat tmp(M,N);
	 /* Build multiplier context for this kernel and store in tmp */
	 pC( _knl_mm.buildKnlIntCtx(srcPos, srcPos, trgPos, tmp) );
	 /* Compute 1.0*tmp*srcDen + 1.0*trgVal */
	 pC( dgemv(1.0, tmp, srcDen, 1.0, trgVal) );
  }/* Break up multiplication into smaller steps */
  else {
	 /* Build sample source positions at the Upward Equivalent location
	  * based on the source node's center srcCtr and radius srcRad */
	 DblNumMat srcPos; pC( _matmgnt->locPos(UE, srcCtr, srcRad, srcPos) );
	 int RUNS = (trgPos.n()-1) / TMAX + 1;
	 for(int r=0; r<RUNS; r++) {
		/* Find start and end locations as well as number of elements being used */
		int stt = r*TMAX;
		int end = min((r+1)*TMAX, trgPos.n());
		int num = end-stt;
		int M = num * _knl_mm.trgDOF();
		int N = srcPos.n() * _knl_mm.srcDOF();
		/* Target positions matrix */
		DblNumMat tps(dim(), num, false, trgPos.data() + stt*dim());
		/* Target values vector */
		DblNumVec tvl(num*_knl_mm.trgDOF(), false, trgVal.data() + stt*_knl_mm.trgDOF());
		DblNumMat tmp(M,N);
		/* Build multiplier context for this kernel and store in tmp */
		pC( _knl_mm.buildKnlIntCtx(srcPos, srcPos, tps, tmp) );
		/* Compute 1.0*tmp*srcDen + 1.0*tvl */
		pC( dgemv(1.0, tmp, srcDen, 1.0, tvl) );
	 }
  }
  return(0);
}

// ----------------------------------------------------------------------
/* For a specific node, return a matrix of exact source positions for this processor as a contributor */
DblNumMat FMM3d_MPI::ctbSrcExaPos(int gNodeIdx) 
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.ctbSrcExaBeg();
  int num = node.ctbSrcExaNum();
  double* arr;     VecGetArray(    _ctbSrcExaPos, &arr);
  double* buf=arr; VecRestoreArray(_ctbSrcExaPos, &arr);
  return DblNumMat(dim(), num, false, buf+beg*dim());
}
/* For a specific node, return a matrix of exact source normals for this processor as a contributor */
DblNumMat FMM3d_MPI::ctbSrcExaNor(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.ctbSrcExaBeg();
  int num = node.ctbSrcExaNum();
  double* arr;     VecGetArray(    _ctbSrcExaNor, &arr);
  double* buf=arr; VecRestoreArray(_ctbSrcExaNor, &arr);
  return DblNumMat(dim(), num, false, buf+beg*dim());
}
/* For a specific node, return a vector of exact source densities for this processor as a contributor */
DblNumVec FMM3d_MPI::ctbSrcExaDen(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.ctbSrcExaBeg();
  int num = node.ctbSrcExaNum();
  double* arr;     VecGetArray(    _ctbSrcExaDen, &arr);
  double* buf=arr; VecRestoreArray(_ctbSrcExaDen, &arr);
  return DblNumVec(srcDOF()*num, false, buf+beg*srcDOF());
}
/* For a specific node, return a vector of contributor source upward equivalent densities */
DblNumVec FMM3d_MPI::ctbSrcUpwEquDen(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int idx = node.ctbSrcNodeIdx();
  double* arr;     VecGetArray(    _ctbSrcUpwEquDen, &arr);
  double* buf=arr; VecRestoreArray(_ctbSrcUpwEquDen, &arr);
  return DblNumVec(datSze(UE), false, buf+idx*datSze(UE));
}
/* For a specific node, return a vector of contributor source upward check values*/
DblNumVec FMM3d_MPI::ctbSrcUpwChkVal(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int idx = node.ctbSrcNodeIdx();
  double* arr;     VecGetArray(    _ctbSrcUpwChkVal, &arr);
  double* buf=arr; VecRestoreArray(_ctbSrcUpwChkVal, &arr);
  return DblNumVec(datSze(UC), false, buf+idx*datSze(UC));
}
// ----------------------------------------------------------------------
/* For a specific node, return matrix of exact source positions for which this processor is a user */
DblNumMat FMM3d_MPI::usrSrcExaPos(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.usrSrcExaBeg();
  int num = node.usrSrcExaNum();
  double* arr;     VecGetArray(    _usrSrcExaPos, &arr);
  double* buf=arr; VecRestoreArray(_usrSrcExaPos, &arr);
  return DblNumMat(dim(), num, false, buf+beg*dim());
}
/* For a specific node, return matrix of exact source normals for which this processor is a user */
DblNumMat FMM3d_MPI::usrSrcExaNor(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.usrSrcExaBeg();
  int num = node.usrSrcExaNum();
  double* arr;     VecGetArray(    _usrSrcExaNor, &arr);
  double* buf=arr; VecRestoreArray(_usrSrcExaNor, &arr);
  return DblNumMat(dim(), num, false, buf+beg*dim());
}
/* For a specific node, return vector of exact source normals for which this processor is a user */
DblNumVec FMM3d_MPI::usrSrcExaDen(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.usrSrcExaBeg();
  int num = node.usrSrcExaNum();
  double* arr;     VecGetArray(    _usrSrcExaDen, &arr);
  double* buf=arr; VecRestoreArray(_usrSrcExaDen, &arr);
  return DblNumVec(srcDOF()*num, false, buf+beg*srcDOF());
}
/* For a specific node, return upward equivalent source densities for which this processor is a user */
DblNumVec FMM3d_MPI::usrSrcUpwEquDen(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int idx = node.usrSrcNodeIdx();
  double* arr;     VecGetArray(    _usrSrcUpwEquDen, &arr);
  double* buf=arr; VecRestoreArray(_usrSrcUpwEquDen, &arr);
  return DblNumVec(datSze(UE), false, buf+idx*datSze(UE));
}
// ----------------------------------------------------------------------
/* For a specific node, return matrix of exact target positions for which this processor is an evaluator */
DblNumMat FMM3d_MPI::evaTrgExaPos(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.evaTrgExaBeg();
  int num = node.evaTrgExaNum();
  double* arr;     VecGetArray(    _evaTrgExaPos, &arr);
  double* buf=arr; VecRestoreArray(_evaTrgExaPos, &arr);
  return DblNumMat(dim(), num, false, buf+beg*dim());
}
/* For a specific node, return vector of exact target values for which this processor is an evaluator */
DblNumVec FMM3d_MPI::evaTrgExaVal(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int beg = node.evaTrgExaBeg();
  int num = node.evaTrgExaNum();
  double* arr;     VecGetArray(    _evaTrgExaVal, &arr);
  double* buf=arr; VecRestoreArray(_evaTrgExaVal, &arr);
  return DblNumVec(trgDOF()*num, false, buf+beg*trgDOF());
}
/* For a specific node, return vector of downward equivalent target densities for which this processor is an evaluator */
DblNumVec FMM3d_MPI::evaTrgDwnEquDen(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int idx = node.evaTrgNodeIdx();
  double* arr;     VecGetArray(    _evaTrgDwnEquDen, &arr);
  double* buf=arr; VecRestoreArray(_evaTrgDwnEquDen, &arr);
  return DblNumVec(datSze(DE), false, buf+idx*datSze(DE));
}
/* For a specific node, return vector of downward check target values for which this processor is an evaluator */
DblNumVec FMM3d_MPI::evaTrgDwnChkVal(int gNodeIdx)
{
  Let3d_MPI::Node& node = _let->node(gNodeIdx);
  int idx = node.evaTrgNodeIdx();
  double* arr;     VecGetArray(    _evaTrgDwnChkVal, &arr);
  double* buf=arr; VecRestoreArray(_evaTrgDwnChkVal, &arr);
  return DblNumVec(datSze(DC), false, buf+idx*datSze(DC));
}


