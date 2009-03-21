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
/*! \file */
#ifndef _SVDREP_HPP_
#define _SVDREP_HPP_

#include "nummat.hpp"

using std::endl;

//! SVD - Singular Value Decomposition Rrepresentation of some input matrix
/*! The SVD Decomposition of a matrix, A, can be written as U*S*V^T where S is
 * a diagonal matrix of singular values, U and V are orthogonal matrices.  This decompition
 * allows for fast solving a a system of linear equations.  Here, the Lapack Fortran
 * code for DGESVD is called.  See lapack.h for more information */
class SVDRep
{
public:
  /*! Construct SVDRep  */
  SVDRep()   {}
  /*! Construct SVDRep  from an existing SVDRep*/
  SVDRep(const SVDRep& C): _matU(C._matU), _matS(C._matS), _matVT(C._matVT)  {}
  /*! Destroy SVDRep */
  ~SVDRep()  {}

  /*! Overloaded "=" operator.  Set this equal to another SVDRep */
  SVDRep& operator=(const SVDRep& c)  { _matU = c._matU; _matS = c._matS; _matVT = c._matVT; return *this; }
  //access
  /*! Returns the matrix _matU from the decomposition U*S*V^T */
  DblNumMat& U() { return _matU; }
  /*! Returns the vector _matS from the decomposition U*S*V^T : S is a matrix of diagonal values only, so is represented as a vector */
  DblNumVec& S() { return _matS; }
  /*! Returns the matrix _matVT from the decomposition U*S*V^T */
  DblNumMat& VT(){ return _matVT; }
  //ops
  /*! Construct the SVD representation of a matrix M with cutoff epsilon.  The smallest singular values
	* are dropped based on epsilon.  The Lapack routine DGESVD from lapack.h is called here as well */
  int construct(double epsilon, const DblNumMat& M);
  /*! Perform the operation:
	*
	*  y = alpha*M*X + beta*Y where M is the original matrix from the SVDRep.
	*
	* Using the SVD representation, this dgemv operation is performed more quickly.
	* This also uses the Blas DGEMV opaeration seen in blas.h
	*/
  int dgemv(double alpha, const DblNumVec& X, double beta, DblNumVec& Y, double tol=0.0); // y <- a Mx + b y
  /*! Perform the operation:
	*
	*  y = alpha*M*X + beta*Y where M is the original matrix from the SVDRep.
	*
	* Using the SVD representation, this dgemv operation is performed more quickly.
	* This also uses the Blas DGEMV opaeration seen in blas.h
	*/
  int dgemv(double alpha, double* X, double beta, double* Y, double tol=0.0);

  /*! m() = the "m" size of U, or _matU.m() */
  int m() const { return _matU.m(); }
  /*! k() is the length of the diagonal of S, or matS.m() */
  int k() const { return _matS.m(); } //length
  /*! n() = the "n" size of VT, or _matVT.n() */
  int n() const { return _matVT.n(); }
  
protected:
  /*! The matrix _matU from the decomposition U*S*V^T */
  DblNumMat _matU;
  /*! The vector _matS from the decomposition U*S*V^T : S is a matrix of diagonal values only, so is represented as a vector */
  DblNumVec _matS;
  /*! The matrix _matVT from the decomposition U*S*V^T */
  DblNumMat _matVT;
};	
/*! Overloaded output operator.  Outputs the sizes and matrices of SVDRep */
inline ostream& operator<<( ostream& os, SVDRep& svdrep)
{
  os<<svdrep.U().m()<<" "<<svdrep.S().m()<<" "<<svdrep.VT().n()<<endl;
  os<<svdrep.U()<<svdrep.S()<<svdrep.VT()<<endl;
  return os;
}

//int matvec(double a, const DblNumVec& X, double b, DblNumVec& Y); // y <- a Mx + b y
//int matvec(double a, double* X, double b, double* Y);

#endif
