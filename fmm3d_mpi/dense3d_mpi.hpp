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
 * Header file for Dense3d class, which provides for a direct dense/non-FMM solver.
 * This code can be used to test the accuracy of the FMM code as well as for debugging purposes.
 */

#ifndef _DENSE3D_HPP_
#define _DENSE3D_HPP_

#include "knlmat3d_mpi.hpp"

using std::vector;

//! Dense3d_MPI implements setup and evaluate from KnlMat3d_MPI  the solver is a "dense" solver - meaning that  fmm is not used, and the solution is "exact".
class Dense3d_MPI: public KnlMat3d_MPI
{
protected:
  //COMPONENTS
  /*! all source positions */
  Vec _srcAllPos;
  /*! all source normals */
  Vec _srcAllNor;
public:
  /*! construct Dense3d_MPI using string prefix */
  Dense3d_MPI(const string& p);
  /*! call ~KnlMat3d_MPI -> ~ComObject_MPI.hpp */
  ~Dense3d_MPI();
  //SETUP AND USE
  /*! Setup makes sure that all processor have the source  positions and noremals scattered to them */
  int setup();
  /*! Evaluate target values based on source densities and positions using direct evaluation */
  int evaluate(Vec srcden, Vec trgval);
};

#endif
