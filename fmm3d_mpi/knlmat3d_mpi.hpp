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
#ifndef _KNLMAT3D_HPP_
#define _KNLMAT3D_HPP_

#include "comobject_mpi.hpp"
#include "common/vec3t.hpp"
#include "kernel3d_mpi.hpp"

using std::pair;

//----------------------------------------------------------------------------------
//! KnlMat3d_MPI is an mpi version of KnlMat3d - storing kernel and matrix information for 3d.  KnlMat3d_MPI builds upon ComObject_MPI - for use with FMM_3d_MPI or Dense3d_MPI by including virtual functions for setup() and evaluate(...) as well as accesses to information such as source positions, target values, etc.
class KnlMat3d_MPI: public ComObject_MPI
{
protected:
  //---PARAMS (REQ)
  /*! Source positions */
  Vec _srcPos;
  /*! Source normals */
  Vec _srcNor;
  /*! Target positions */
  Vec _trgPos;
  /*! the kernel being used, defined in kernel3d_mpi.hpp - defines kernel type and coefficients */
  Kernel3d_MPI _knl;
public:
  /*! Construct the KnlMat3d_MPI object, using a string, which is passed to ComObject.
	* For example, "fmm3d_" for fmm3d.  _srcPos, _srcNor, and _trgPos are set to NULL */
  KnlMat3d_MPI(const string& p): 
	 ComObject_MPI(p), _srcPos(NULL), _srcNor(NULL), _trgPos(NULL) {;}
  /*! destructor - calls ~ComObject() */
  virtual ~KnlMat3d_MPI() { }
  //MEMBER ACESS
  /*! Return source positions */
  Vec& srcPos() { return _srcPos; }
  /*! Return source normals */
  Vec& srcNor() { return _srcNor; }
  /*! Return target positions */
  Vec& trgPos() { return _trgPos; }
  /* Return kernel (kernel type and coefficients) */
  Kernel3d_MPI& knl()    { return _knl; }
  //SETUP and USE
  /*! virtual setup function */
  virtual int setup()=0;
  /*! virtual evaluation function.
	* takes source densities and evaluates target values depending on kernel */
  virtual int evaluate(Vec srcDen, Vec trgVal) = 0;
  //OTHER ACCESS
  /*! Return dimension of the problem.  Set to 3 */
  int dim() { return 3; }
  /*! Return source degree of freedom: _knl.srcDOF() - see kernel3d_mpi.hpp */
  int srcDOF() { return _knl.srcDOF(); }
  /*! Return target degree of freedom: _knl.trgDOF() - see kernel3d_mpi.hpp */
  int trgDOF() { return _knl.trgDOF(); }
  //EXTRA STUFF
  /*! local number of positions for this processor */
  PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/dim(); }
  /*! global number of positions for this processor */
  PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/dim(); }
  /*! processor Local Range.
	 for vector pos, return the beginning and end range positions for which the local processor is responsible */
  void procLclRan(Vec pos, PetscInt& beg, PetscInt& end) { VecGetOwnershipRange(pos, &beg, &end); beg=beg/dim(); end=end/dim(); }
};

#endif


