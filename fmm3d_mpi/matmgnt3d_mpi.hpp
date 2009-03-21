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
#ifndef _MATMGNT3D_HPP_
#define _MATMGNT3D_HPP_

#include "common/nummat.hpp"
#include "common/numtns.hpp"
#include "common/offtns.hpp"
#include "common/vec3t.hpp"
#include "kernel3d_mpi.hpp"
#include "comobject_mpi.hpp"

using std::map;
using std::pair;

//--------------------------------------
//unique identifier: equation
//! Matrix Magament MPI implementation.  Information for up/down equivalent/check surfaces contained here as well as how to transition and multiply between them.  For use in fmm3d_mpi_eval.cpp in the FMM3d_MPI::evaluate(.,.) function.  More information available.
/*!
 */
class MatMgnt3d_MPI
{
public:
  enum {	 /*! Upward Equivalent */ UE=0,
			 /*! Upward Check */ UC=1,
			 /*! Downward Equivalent */ DE=2,
			 /*! Downward Check */ DC=3,  };
public:
  /*! Provided by fmm3d_mpi, the kernel environment for doing operations.  Allows for kernel independence in multiplications */
  Kernel3d_MPI _knl; 
  /*! Degree of precision */
  int _np;		
  /*! Homogeneous or not */
  bool _homogeneous;
  vector<double> _degVec;
  //COMPONENTS
  /*! Upward check to upward equivalent map */
  map<int, DblNumMat> _upwChk2UpwEqu;
  /*! upward equivalent to upward check map */
  map<int, NumTns<DblNumMat> > _upwEqu2UpwChk;
  /*! downward check to downward equivalent */
  map<int, DblNumMat> _dwnChk2DwnEqu;
  /*! downward equivalent to downward check */
  map<int, NumTns<DblNumMat> > _dwnEqu2DwnChk;
  /*! upward equivalent to downward check */
  map<int, OffTns<DblNumMat> > _upwEqu2DwnChk;
  /*! sample positions.  This is an array of 4 matrices, for each of UE, UC, DE and DC */
  DblNumMat _samPos[4];
  /*! regular positions */
  DblNumMat _regPos; 
  
public:
  MatMgnt3d_MPI();
  ~MatMgnt3d_MPI();
  //MEMBER ACCESS
  /*! Return 3d kernel */
  Kernel3d_MPI& knl() { return _knl; }
  /*! Return degree of precision in _np */
  int& np() { return _np; }
  /*! Return 0.1^(_np+1) - tolerance level in computing pseudoinverse */
  double alt(); //TODO: decide it based on np

  /*! Return source degree of freedom */
  int srcDOF() { return _knl.srcDOF(); }
  /*! Return target degree of freedom */
  int trgDOF() { return _knl.trgDOF(); }
  /*! Return problem dimension.  Equals 3 */
  int dim() { return 3; }
  //SETUP AND USE
  /*! setup() calculates sample positions for up/down quivalent/check as well as calculates regular positions based on _np (degree of precision) */
  int setup();
  /*! Report sizes of different maps */
  int report();
  /*! Size of the plain data */
  int plnDatSze(int tp);
  /*! Size of the effective data (when using FFTs) */
  int effDatSze(int tp);

  /*! Upward Check to Upward Equivalent Multiplication */
  int UpwChk2UpwEqu_dgemv(int level,             const DblNumVec&, DblNumVec&);
  /*! Upward Equivalent to Upward Check Multiplication */
  int UpwEqu2UpwChk_dgemv(int level, Index3 ii, const DblNumVec&, DblNumVec&);
  /*! Downward Check to Downward Equivalent Multiplcation */
  int DwnChk2DwnEqu_dgemv(int level,             const DblNumVec&, DblNumVec&);
  /*! Downward Equivalent to Downward Check Multiplcation */
  int DwnEqu2DwnChk_dgemv(int level, Index3 ii, const DblNumVec&, DblNumVec&);
  /*! Upward Equivalent to Downward Check Mutiplcation */
  int UpwEqu2DwnChk_dgemv(int level, Index3 ii, const DblNumVec& effDen, DblNumVec& effVal);

  /*! Convert plain density data to regular to effective density data for use with FFTs */
  int plnDen2EffDen(int level, const DblNumVec&, DblNumVec&);
  /*! Convert sample density data to regular density data */
  int samDen2RegDen(const DblNumVec&, DblNumVec&);
  /*! Convert effective values to regular to plain values */
  int effVal2PlnVal(int level, const DblNumVec&, DblNumVec&);
  /*! Convert regular values to sample values */
  int regVal2SamVal(const DblNumVec&, DblNumVec&);

  /*! Return the sample positions for tp (UE, UC, DE, or DC) */
  const DblNumMat& samPos(int tp) { return _samPos[tp]; }
  /*! Return regular positions */
  const DblNumMat& regPos()       { return _regPos; }
  /*! local positions - first argument is UC, DE, DC, UC.  Returns local positions based on sample positions requested*/
  int locPos(int, Point3, double, DblNumMat&);
  /*! Calculate various grids for sample positions (UE, UC, DE, and DC grid ) */
  int samPosCal(int n, double R, DblNumMat& ret);
  /*! Calculate FFT regular position grid */
  int regPosCal(int n, double R, DblNumMat& ret); 
  int cptwvv(int, double, fftw_complex*, int, fftw_complex*, int, fftw_complex*, int);
  
public:
  static double _wsbuf[];
  /*! Vector of matrix managers.  Stores previous MatMgnt3d_MPIs if they exist */
  static vector<MatMgnt3d_MPI> _mmvec;
public:
  /*! Get a pointer to a matrix manager for this kernel if it exists
	* If not, create one and return that pointer */
  static MatMgnt3d_MPI* getmmptr(Kernel3d_MPI, int);  
};

#endif


