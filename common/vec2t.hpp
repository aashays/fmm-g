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
#ifndef  _VEC2T_HPP_
#define  _VEC2T_HPP_

#include "commoninc.hpp"

using std::istream;
using std::ostream;
using std::min;
using std::max;
using std::abs;

//-----------------------------------------------------------------------------------------------------
//!Common 2D VECtor Template
template <class F>
//! Common 2D Vector template-based class
class Vec2T {
private:
  /*! 2D vector Data of type F */
  F _v[2];
public:
  enum{ X=0, Y=1 };
  //------------CONSTRUCTOR AND DESTRUCTOR
  /*! Constructor */
  Vec2T()              { _v[0]=F(0);    _v[1]=F(0); }
  /*! Constructor.  Set _v[0], _v[1] = f of type F */
  Vec2T(F f)           { _v[0]=f;       _v[1]=f; }
  /*! Constructor.  Set _v[0], _v[1] = f of type F */
  Vec2T(const F* f)    { _v[0]=f[0];    _v[1]=f[1]; }
  /*! Constructor.  Set _v[0] = a _v[1] = b of type F */
  Vec2T(F a,F b)       { _v[0]=a;       _v[1]=b; }
  /*! Constructor.  Set _v[0] = c._v[0], _v[1] = c._v[1] */
  Vec2T(const Vec2T& c){ _v[0]=c._v[0]; _v[1]=c._v[1]; }
  /*! Destroy vector */
  ~Vec2T() {}
  //------------POINTER and ACCESS
  /*! Return pointer to _v[0] */
  operator F*()             { return &_v[0]; }
  /*! Return pointer to _v[0] */
  operator const F*() const { return &_v[0]; }
  /*! Return pointer to _v[0] */
  F* array()                { return &_v[0]; }  //access array
  /*! Return ith (0 or 1) element of _v */
  F& operator()(int i)             { assert(i<2); return _v[i]; }
  /*! Return ith (0 or 1) element of _v */
  const F& operator()(int i) const { assert(i<2); return _v[i]; }
  /*! Return ith (0 or 1) element of _v */
  F& operator[](int i)             { assert(i<2); return _v[i]; }
  /*! Return ith (0 or 1) element of _v */
  const F& operator[](int i) const { assert(i<2); return _v[i]; }
  /*! Return x-coordinate (_v[0]) */
  F& x()             { return _v[0];}
  /*! Return y-coordinate (_v[1]) */
  F& y()             { return _v[1];}
  /*! Return x-coordinate (_v[0]) */
  const F& x() const { return _v[0];}
  /*! Return y-coordinate (_v[1]) */
  const F& y() const { return _v[1];}
  //------------ASSIGN
  /*! Overloaded = operator.  this._v[0] = c._v[0], etc. */
  Vec2T& operator= ( const Vec2T& c ) { _v[0] =c._v[0]; _v[1] =c._v[1]; return *this; }
  /*! Overloaded += operator.  this._v[0] += c._v[0], etc. */
  Vec2T& operator+=( const Vec2T& c ) { _v[0]+=c._v[0]; _v[1]+=c._v[1]; return *this; }
  /*! Overloaded -= operator.  this._v[0] -= c._v[0], etc. */
  Vec2T& operator-=( const Vec2T& c ) { _v[0]-=c._v[0]; _v[1]-=c._v[1]; return *this; }
  /*! Overloaded *= operator.  this._v[0] *= c._v[0], etc. */
  Vec2T& operator*=( const F& s )     { _v[0]*=s;       _v[1]*=s;       return *this; }
  /*! Overloaded /= operator.  this._v[0] /= c._v[0], etc. */
  Vec2T& operator/=( const F& s )     { _v[0]/=s;       _v[1]/=s;       return *this; }
  //-----------LENGTH...
  /*! L-1 norm:  sabsolute value of sum of elements in vector */
  F l1( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+abs(_v[i]); return sum; }
  /*! L-infinity norm:  max of elements in vector */
  F linfty( void ) const  { F cur=F(0); for(int i=0; i<2; i++) cur=max(cur,abs(_v[i])); return cur; }
  /*! L-2 norm:  square root of sum of elements */
  F l2( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+_v[i]*_v[i]; return sqrt(sum); }
  /*! Length = L-2 norm */
  F length( void ) const  { return l2(); }
  /*! Unit vector in director of this */
  Vec2T dir( void )    const  { F a=l2(); return (*this)/a; }
};

//-----------BOOLEAN OPS
/*! Boolean == overloaded operator returns true if a==b */
template <class F> inline bool operator==(const Vec2T<F>& a, const Vec2T<F>& b) {
  bool res = true;  for(int i=0; i<2; i++)   res = res && (a(i)==b(i));  return res;
}
/*! Boolean != overloaded operator returns true if a!=b */
template <class F> inline bool operator!=(const Vec2T<F>& a, const Vec2T<F>& b) {
  return !(a==b);
}
/*! Boolean > overloaded operator returns true if a > b in BOTH x and y-directions */
template <class F> inline bool operator> (const Vec2T<F>& a, const Vec2T<F>& b) {
  bool res = true;  for(int i=0; i<2; i++)   res = res && (a(i)> b(i));  return res; 
}
/*! Boolean < overloaded operator returns true if a < b in BOTH x and y-directions */
template <class F> inline bool operator< (const Vec2T<F>& a, const Vec2T<F>& b) {
  bool res = true;  for(int i=0; i<2; i++)   res = res && (a(i)< b(i));  return res; 
}
/*! Boolean >= overloaded operator returns true if a >= b in BOTH x and y-directions */
template <class F> inline bool operator>=(const Vec2T<F>& a, const Vec2T<F>& b) {
  bool res = true;  for(int i=0; i<2; i++)	res = res && (a(i)>=b(i));  return res; 
}
/*! Boolean <= overloaded operator returns true if a <= b in BOTH x and y-directions */
template <class F> inline bool operator<=(const Vec2T<F>& a, const Vec2T<F>& b) {
  bool res = true;  for(int i=0; i<2; i++)   res = res && (a(i)<=b(i));  return res; 
}

//-----------NUMERICAL OPS
/*! Overloaded "-" numerical operation.  Return negation of vector a */
template <class F> inline Vec2T<F> operator- (const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = -a[i]; return r;
}
/*! Overloaded "+" numerical operation.  Return addition of components of a and b */
template <class F> inline Vec2T<F> operator+ (const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]+b[i]; return r; 
}
/*! Overloaded "-" numerical operation.  Return subtraction of compoenets of b from a */
template <class F> inline Vec2T<F> operator- (const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]-b[i]; return r;
}
/*! Overloaded "*" numerical operation.  Return scaling of components of a by scl */
template <class F> inline Vec2T<F> operator* (F scl, const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = scl*a[i];  return r;
}
/*! Overloaded "*" numerical operation.  Return scaling of components of a by scl */
template <class F> inline Vec2T<F> operator* (const Vec2T<F>& a, F scl) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = scl*a[i];  return r;
}
/*! Overloaded "/" numerical operation.  Return scaling of components of a by 1/scl */
template <class F> inline Vec2T<F> operator/ (const Vec2T<F>& a, F scl) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]/scl;  return r;
}
/*! Overloaded "*" numerical operation.  Return a[0]*b[0] + a[1]*b[1] */
template <class F> inline F operator* (const Vec2T<F>& a, const Vec2T<F>& b) {
  F sum=F(0); for(int i=0; i<2; i++) sum=sum+a(i)*b(i); return sum;
}
/*! Dot-product.  Return a*b = a[0]*b[0] + a[1]*b[1] */
template <class F> inline F dot       (const Vec2T<F>& a, const Vec2T<F>& b) {
  return a*b;
}
//-------------ew OPS
/*! Return r where r[0] = min(a[0],b[0]), etc. */
template <class F> inline Vec2T<F> min(const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = min(a[i], b[i]); return r;
}
/*! Return r where r[0] = max(a[0],b[0]), etc. */
template <class F> inline Vec2T<F> max(const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = max(a[i], b[i]); return r;
}
/*! Return r where r[0] = abs(a[0]), etc. */
template <class F> inline Vec2T<F> abs(const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = abs(a[i]); return r;
}
/*! Return r where r[0] = a[0]*b[0], etc. */
template <class F> inline Vec2T<F> ewmul(const Vec2T<F>&a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]*b[i]; return r;
}
/*! Return r where r[0] = a[0]/b[0], etc. */
template <class F> inline Vec2T<F> ewdiv(const Vec2T<F>&a, const Vec2T<F>& b) { 
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]/b[i]; return r;
}
//---------------INOUT
/*! input >> operator.  is >> a[i] */
template <class F> istream& operator>>(istream& is, Vec2T<F>& a) {
  for(int i=0; i<2; i++) is>>a[i]; return is;
}
/*! output << operator.  os << a[i] */
template <class F> ostream& operator<<(ostream& os, const Vec2T<F>& a) { 
  for(int i=0; i<2; i++) os<<a[i]<<" "; return os;
}

//---------------------------------------------------------
/// MOST COMMONLY USED
/*! A 2-D point with the x and y-coordinates represented as doubles */
typedef Vec2T<double> Point2;
/*! A 2-D index with the two indices represented as integers */ 
typedef Vec2T<int>    Index2;



#endif
