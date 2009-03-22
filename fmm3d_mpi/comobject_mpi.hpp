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
#ifndef _COMOBJECT_HPP_
#define _COMOBJECT_HPP_

#include "common/commoninc.hpp"
#include "petscsnes.h"

using std::string;
using std::map;

//! ComObject_MPI class for communication purposes.  ComObject stores basic message passing interface  (mpi) information such as size and rank  as defined in files included from using petsc */
class ComObject_MPI
{
protected:
  /*! _prefix is a string such as "fmm3d_" which describes the type of object this is */
  string _prefix;
public:
  /*! Initialize ComObject using string prefix */
  ComObject_MPI(const string& prefix): _prefix(prefix) {;}
  virtual ~ComObject_MPI() {;}
  //-------------------------
  /*! Return _prefix string */
  const string& prefix() { return _prefix; }
  /*! return MPI_comm - set to PETSC_COMM_WORLD */
  const MPI_Comm& mpiComm( ) const { return PETSC_COMM_WORLD; }
  /*! return rank (id) of MPI_Comm_rank of PETSC_COMM_WORLD */
  int mpiRank() const { int rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); return rank; }
  /*! return size of MPI_Comm_size of PETSC_COMM_WORLD */
  int mpiSize() const { int size; MPI_Comm_size(PETSC_COMM_WORLD, &size); return size; }
};

/*! like iC from petsc, allows for error checking from functinos which return certain integer values on error output */
#define pC(fun)  { int ierr=fun; CHKERRQ(ierr); }
/*! like iA from petsc, allows for checking expression failure */
#define pA(expr) { if(!(expr)) SETERRQ(1, "Assertion: "#expr" failed!"); }

#endif
