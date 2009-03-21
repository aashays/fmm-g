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
#ifndef _NUMVEC_HPP_
#define _NUMVEC_HPP_

#include "commoninc.hpp"

using std::ostream;
using std::ios_base;
using std::endl;

//! Class template definition.  F can be any type, mostly used as data for the FMM purposes
template <class F>
//! NumVec is a vector made up of a size and data.  The data can be made up of any type (ints, double, bools, etc.)
class NumVec
{
public:
  /*! _m is the size or length of the vector */
  int  _m;
  /*! Whether or not the vector "owns" its data or just shares a pointer to it */
  bool _owndata;
  /*! the data itself of type F (double, bool, etc.) */
  F* _data;
public:
  /*! Construct a NumVec and create _data array F[_m] */
  NumVec(int m=0): _m(m), _owndata(true)  {
	 if(_m>0) { _data = new F[_m]; assert(_data!=NULL); memset(_data, 0, _m*sizeof(F)); } else _data=NULL;
  }
  /*! Construct NumVec of size _m and whether or not it owns its data (_owndata).  If so, create space for _data = F[_m].
  * Otherwise, point _data to the data input */
  NumVec(int m, bool owndata, F* data): _m(m), _owndata(owndata) {
	 if(_owndata) {
		if(_m>0) { _data = new F[_m]; assert(_data!=NULL); memset(_data, 0, _m*sizeof(F)); } else _data=NULL;
		if(_m>0) memcpy( _data, data, _m*sizeof(F) );
	 } else {
		_data = data;
	 }
  }
  /*! Set this NumVec equal to C */
  NumVec(const NumVec& C): _m(C._m), _owndata(C._owndata)  {
	 if(_owndata) {
		if(_m>0) { _data = new F[_m]; assert(_data!=NULL); memset(_data, 0, _m*sizeof(F)); } else _data=NULL;
		if(_m>0) memcpy( _data, C._data, _m*sizeof(F) );
	 } else {
		_data = C._data;
	 }
  }
  /*! Destroy this NumVec*/
  ~NumVec() {
	 if(_owndata) {
		if(_m>0) { delete[] _data; _data = NULL; }
	 }
  }
  /*! Set this NumVec equal to C using the overloaded "=" operator */ 
  NumVec& operator=(const NumVec& C)  {
	 if(_owndata) { 
		if(_m>0) { delete[] _data; _data = NULL; }
	 }
	 _m = C._m; _owndata=C._owndata;
	 if(_owndata) {
		if(_m>0) { _data = new F[_m]; assert(_data!=NULL); memset(_data, 0, _m*sizeof(F)); } else _data=NULL;
		if(_m>0) memcpy( _data, C._data, _m*sizeof(F) );
	 } else {
		_data =C._data;
	 }
	 return *this;
  }
  /*! resize this NumVec to size input m */
  void resize(int m)  {
	 assert(_owndata==true);
	 if(m !=_m) {
		if(_m>0) { delete[] _data; _data = NULL; }
		_m = m;
		if(_m>0) { _data = new F[_m]; assert(_data!=NULL);  memset(_data, 0, _m*sizeof(F)); } else _data=NULL;
	 }
  }
  /*! return _data[i] of type F */
  const F& operator()(int i) const  {	 assert(i>=0 && i<_m);
    return _data[i]; 
  }
  /*! return _data[i] of type F */
  F& operator()(int i)  {	 assert(i>=0 && i<_m);
    return _data[i]; 
  }
  /*! Return pointer to _data */
  F* data() const { return _data; }
  /*! Return size m */
  int m () const { return _m; }
};
/*! output stream operator << */
template <class F> inline ostream& operator<<( ostream& os, const NumVec<F>& vec)
{
  os<<vec.m()<<endl;
  os.setf(ios_base::scientific, ios_base::floatfield);
  for(int i=0; i<vec.m(); i++)	 os<<" "<<vec(i);
  os<<endl;
  return os;
}
/*! Set all members of a NumVec V equal to val.  That is, V(i) = V._data[i] = val for all i */
template <class F> inline void setvalue(NumVec<F>& V, F val)
{
  for(int i=0; i<V.m(); i++)
	 V(i) = val;
}
/*! Clear all values of NumVec V */
template <class F> inline void clear(NumVec<F>& V)
{
  memset(V.data(), 0, V.m()*sizeof(F));
}
/*! Numvec of bools */
typedef NumVec<bool>   BolNumVec;
/*! Numvec of ints */
typedef NumVec<int>    IntNumVec;
/*! Numvec of doubles */
typedef NumVec<double> DblNumVec; //typedef NumVec<double> SclNumVec;


#endif
