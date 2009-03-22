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
#ifndef _NUMMAT_HPP_
#define _NUMMAT_HPP_

#include "numvec.hpp"

//! Class template definition.  F can be any type, mostly used as data for the FMM purposes
template <class F>
//! NumMat is a matrix made up of a size and data.  The data can be made up of any type (ints, double, bools, etc.)
class NumMat
{
public:
  /*! _m is the "m" dimension, or number of rows in the matrix */
  int _m;
  /*! _n is the "n" dimension, or number of columns in the matrix */
  int _n;
  /*! Whether or not data is own */
  bool _owndata;
  /*! The data.  Since using templates, _data can be of any type */
  F* _data;
public:
  /*! Construct a NumMat using the dimensions, m and n.  owndata set to true */
  NumMat(int m=0, int n=0): _m(m), _n(n), _owndata(true) {
	 if(_m>0 && _n>0) { _data = new F[_m*_n]; assert( _data!=NULL ); memset(_data, 0, _m*_n*sizeof(F)); } else _data=NULL;
  }
  /*! Construct NumMat with data given.  If owndata is false, then the data passed is not stored in a new array, F.
	* Otherwise, an array of type F must be allocated into memory to store the data */
  NumMat(int m, int n, bool owndata, F* data): _m(m), _n(n), _owndata(owndata) {
	 if(_owndata) {
		if(_m>0 && _n>0) { _data = new F[_m*_n]; assert( _data!=NULL ); memset(_data, 0, _m*_n*sizeof(F)); } else _data=NULL;
		if(_m>0 && _n>0) memcpy( _data, data, _m*_n*sizeof(F) );
	 } else {
		_data = data;
	 }
  }
  /*! Construct a new matrix with another matrix as argument.  New matrix gets C._m, C._n and C._owndata.
	* If _owndata is true, the new matrix generates an array F to copy the data.  If _owndata is false, _data is just set
	* to C._data, simply pointing to the same data */
  NumMat(const NumMat& C): _m(C._m), _n(C._n), _owndata(C._owndata) {
	 if(_owndata) {
		if(_m>0 && _n>0) { _data = new F[_m*_n]; assert( _data!=NULL ); memset(_data, 0, _m*_n*sizeof(F)); } else _data=NULL;
		if(_m>0 && _n>0) memcpy( _data, C._data, _m*_n*sizeof(F) );
	 } else {
		_data = C._data;
	 }
  }
  /*! Destroy NumMat.  If _owndata is true, _data must be deleted and deallocated */
  ~NumMat() { 
	 if(_owndata) { 
		if(_m>0 && _n>0) { delete[] _data; _data = NULL; } 
	 }
  }
  /*! A way of setting an existing matrix equal to another exisiting matrix */
  NumMat& operator=(const NumMat& C) {
	 if(_owndata) { 
		if(_m>0 && _n>0) { delete[] _data; _data = NULL; } 
	 }
	 _m = C._m; _n=C._n; _owndata=C._owndata;
	 if(_owndata) {
		if(_m>0 && _n>0) { _data = new F[_m*_n]; assert( _data!=NULL ); memset(_data, 0, _m*_n*sizeof(F)); } else _data=NULL;
		if(_m>0 && _n>0) memcpy( _data, C._data, _m*_n*sizeof(F) );
	 } else {
		_data = C._data;
	 }
	 return *this;
  }
  /*! Resize the matrix with new m and n.  Asserts _owndata is true so that new array of type F can be allocated into memory */
  void resize(int m, int n)  {
	 assert( _owndata==true );
	 if(_m!=m || _n!=n) {
		if(_m>0 && _n>0) { delete[] _data; _data = NULL; } 
		_m = m; _n = n;
		if(_m>0 && _n>0) { _data = new F[_m*_n]; assert( _data!=NULL ); memset(_data, 0, _m*_n*sizeof(F)); } else _data=NULL;
	 }
  }
  /*! Return _data(i,j) or _data[i+j*_m] */ 
  const F& operator()(int i, int j) const  { 
	 assert( i>=0 && i<_m && j>=0 && j<_n );
	 return _data[i+j*_m];
  }
  /*! Return _data(i,j) or _data[i+j*_m] */ 
  F& operator()(int i, int j)  { 
	 assert( i>=0 && i<_m && j>=0 && j<_n );
	 return _data[i+j*_m];
  }
  /*! Return _data */
  F* data() const { return _data; }
  /*! Return _data j*_m */
  F* clmdata(int j) { return &(_data[j*_m]); }
  /*! Return m */
  int m() const { return _m; }
  /*! Return n */
  int n() const { return _n; }
};

/*! Provides a way for streaming out contents of a matrix mat */
template <class F> inline ostream& operator<<( ostream& os, const NumMat<F>& mat)
{
  os<<mat.m()<<" "<<mat.n()<<endl;
  os.setf(ios_base::scientific, ios_base::floatfield);
  for(int i=0; i<mat.m(); i++) {
	 for(int j=0; j<mat.n(); j++)
		os<<" "<<mat(i,j);
	 os<<endl;
  }
  return os;
}
/*! set all of the entries of a NumMat M's _data to a value val.  For example, can set all of a matrix to zero */
template <class F> inline void setvalue(NumMat<F>& M, F val)
{
  for(int i=0; i<M.m(); i++)
	 for(int j=0; j<M.n(); j++)
		M(i,j) = val;
}
/*! Clear all of M's _data values (set to zero) */
template <class F> inline void clear(NumMat<F>& M)
{
  memset(M.data(), 0, M.m()*M.n()*sizeof(F));
}

/*! BolNumMat is a NumMat whose entries are all bools */
typedef NumMat<bool>   BolNumMat;
/*! BolNumMat is a NumMat whose entries are all ints */
typedef NumMat<int>    IntNumMat;
/*! BolNumMat is a NumMat whose entries are all doubles */
typedef NumMat<double> DblNumMat; 

/*
  void getColumn(int j, Vector<F>& vec)  {
  assert( j>=0 && j<n() );
  vec.resize(m());
  for(int i=0; i<m(); i++)
  vec(i) = (*this)(i,j);
  }
  void getRow(int i, Vector<F>& vec)  {
  assert( i>=0 && i<m() );
  vec.resize(n());
  for(int j=0; j<n(); j++)
  vec(j) = (*this)(i,j);
  }
  void setColumn(int j, Vector<F>& vec)  {
  assert( j>=0 && j<n() );
  assert( vec.length() == m() );
  for(int i=0; i<m(); i++)
  (*this)(i,j) = vec(i);
  }
  void setRow(int i, Vector<F>& vec)  {
  assert( i>=0 && i<m());
  assert( vec.length() == n());
  for(int j=0; j<n(); j++)
  (*this)(i,j) = vec(j);
  }
*/



#endif




