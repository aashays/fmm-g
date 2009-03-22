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
 * This file allows for the kernel independence by constructing
 * multiplier/interpolation contexts based on the available kernels.
 * All available kernels are listed here in kernel3d_mpi.hpp and their
 * implementations in kernel3d_mpi.cpp
 */

#ifndef _KERNEL3D_HPP_
#define _KERNEL3D_HPP_

#include "common/nummat.hpp"

using std::vector;

//eqt: 1 2 3 4 5 6
//lyr: s d r p
//qnt: u p ...

enum {
  //laplace kernels
  /*! Laplacian single-layer velocity kernel */
  KNL_LAP_S_U = 111,
  /*! Laplacian double-layer velocity kernel */
  KNL_LAP_D_U = 121,
  /*! Laplacian identity tensor */
  KNL_LAP_I   = 191, 
  //stokes kernels
  /*! Stokes - Fmm3d velocity kernel */
  KNL_STK_F_U = 301,
  /*! Stokes single-layer velocity */
  KNL_STK_S_U = 311,
  /*! Stokes double-layer velocity */
  KNL_STK_S_P = 312,
  /*! Stokes single-layer pressure */
  KNL_STK_D_U = 321,
  /*! Stokes double-layer pressure */
  KNL_STK_D_P = 322,
  /*! Stokes R velocity */
  KNL_STK_R_U = 331,
  /*! Stokes R pressure */
  KNL_STK_R_P = 332,
  /*! Stokes Identity */
  KNL_STK_I   = 391,
  /*! Levi-Civita Tensor */
  KNL_STK_E   = 392, 
  //navier kernels
  /*! Navier single-layer velocity */
  KNL_NAV_S_U = 511,
  /*! Navier double-layer velocity */
  KNL_NAV_D_U = 521,
  /*! Navier R velocity */
  KNL_NAV_R_U = 531,
  /*! Navier identity tensor */
  KNL_NAV_I   = 591,
  /*! Levi-Civita Tensor */
  KNL_NAV_E   = 592, 
  //other kernels
  /*! Error kernel */  
  KNL_ERR = -1
};

//! static class Kernel3d_MPI for 3d kernels of various types for parallel fmm3d code, fmm3d_mpi 
class Kernel3d_MPI
{
protected:
  /*! kernel type is based on the enumerators.  Example, KNL_NAV_S_U = navier single velocity/displacement kernel = 511 */
  int _kernelType;
  /*! coefficients for kernel - for use with Navier and Stokes kernels (example mu viscosity).  See papers for more details */
  vector<double> _coefs;
  /*! minimal difference */
  static double _mindif; 
public:
  /*! Initialize Kernel3d_MPI to default KNL_ERR */
  Kernel3d_MPI(): _kernelType(KNL_ERR) {;}
  /*! Initialize kernel3d_MPI based on kernelType and coefficients for the kernel */
  Kernel3d_MPI(int kernelType, const vector<double>& coefs): _kernelType(kernelType), _coefs(coefs) {;}
  /*! Initialize kernel3d_MPI based on another kernel */
  Kernel3d_MPI(const Kernel3d_MPI& c): _kernelType(c._kernelType), _coefs(c._coefs) {;}
  /*! Set a Kernel3d_MPI using "=" sign */
  Kernel3d_MPI& operator=(const Kernel3d_MPI& c) {
	 _kernelType = c._kernelType; _coefs = c._coefs; return *this;
  }
  /*! kernel type returns which kernel is being used, based on enumeration of kernel types (i.e.KNL_LAP_S_U = 111).  Defaults to KNL_ERR if undefined */
  int& kernelType() { return _kernelType; }
  /*! kernel type returns which kernel is being used, based on enumeration of kernel types (i.e.KNL_LAP_S_U = 111).  Defaults to KNL_ERR if undefined */
  const int& kernelType() const { return _kernelType; }
  /*! Return vector of coefficients used for kernel computation */
  vector<double>& coefs() { return _coefs; }
  /*! Return vector of coefficients used for kernel computation */
  const vector<double>& coefs() const { return _coefs; }
  /*! Dimension of the problem, set to 3 */
  int dim() { return 3; }
  /*! Degree of freedom for source information */
  int srcDOF() const;
  /*! Degree of freedom for target information.  Example is that dof is 1 for pressure, higher for velocity/displacement, based on dimension of problem and kernelType*/
  int trgDOF() const;
  /*! homogeneous or not */
  bool homogeneous() const;
  /*! homogeneous degree, vector size == srcDOF */
  void homogeneousDeg(vector<double>&) const; 
  /*! Each kernelType handles coefs in its own way, can be empty
	* In general, this function, depending on kernelType, builds an
	* interpolation context for the kernel.  For example, (1/4pi)*(1/r^3) for
	* single-layer Laplacian Velocity/Position
	*/
  int buildKnlIntCtx(const DblNumMat& srcPos, const DblNumMat& srcNor, const DblNumMat& trgPos, DblNumMat& inter);
};

/*! inline == boolean operator determines equality based on whether kernelType's for kernel a and b are the same as well as whether the coefficients are equal as well */
inline bool operator==(const Kernel3d_MPI& a, const Kernel3d_MPI& b) {
  return (a.kernelType()==b.kernelType() && a.coefs()==b.coefs());
}



#endif
