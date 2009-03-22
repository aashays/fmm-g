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

#ifndef _COMMONINC_HPP_
#define _COMMONINC_HPP_

//STL stuff
#include <iostream>
#include <fstream>
#include <sstream>

#include <cfloat>
#include <cassert>
#include <cmath>
#include <string>
#include <complex>

#include <vector>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <utility>
#include <algorithm>

//FFTW stuff
#include "fftw3.h"

//BLAS LAPACK STUFF
#include "blas.h"
#include "lapack.h"

/*! Complex number */
typedef std::complex<double> cpx;

//AUX functions
/*! Power function:  2^l.  Uses shifts */
inline int pow2(int l) { assert(l>=0); return (1<<l); }

/*! iC asserts that a function returns 0 on completion */
#define iC(fun)  { int ierr=fun; assert(ierr==0); }
/*! iA asserts that an expression is not false - returns "wrong" if so */
#define iA(expr) { if((expr)==0) { std::cerr<<"wrong"<<std::endl; assert(expr); } } //assert

#endif
