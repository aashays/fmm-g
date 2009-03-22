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
 * Implementation of dense or "exact" multiplication.
 * This code can be used to test the accuracy of the FMM code as well as for debugging purposes.
 */

#include "dense3d_mpi.hpp"
#include "common/vecmatop.hpp"


/* Constructor for Dense3d_MPI - takes prefix string as argument*/
Dense3d_MPI::Dense3d_MPI(const string& p): 
  KnlMat3d_MPI(p), _srcAllPos(NULL), _srcAllNor(NULL)
{
}
/* Destructor for Dense3d_MPI */
Dense3d_MPI::~Dense3d_MPI()
{
  if(_srcAllPos!=NULL) {	 VecDestroy(_srcAllPos);  }
  if(_srcAllNor!=NULL) {	 VecDestroy(_srcAllNor);  }
}

// ----------------------------------------------------------------------
/* Dense3d_MPI setup function scatters source positions and normals
 * as necessary */
#undef __FUNCT__
#define __FUNCT__ "Dense3d_MPI::setup"
int Dense3d_MPI::setup()
{
  //begin
  pA(_srcPos!=NULL && _srcNor!=NULL && _trgPos!=NULL);
  //--------------------------------------------------------------------------
  /* Create a Scatter context - copies all source position values to each processor
	* See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecScatterCreateToAll.html */
  {
	 VecScatter ctx;
	 pC( VecScatterCreateToAll(_srcPos, &ctx, &_srcAllPos) );
	 // VecScatterBegin(VecScatter inctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
	 pC( VecScatterBegin(ctx, _srcPos, _srcAllPos, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterEnd(ctx,  _srcPos, _srcAllPos, INSERT_VALUES, SCATTER_FORWARD ) );
	 pC( VecScatterDestroy(ctx) );
  }
  /* Create a Scatter context - copies all source normal values to each processor */
  {
	 VecScatter ctx;
	 pC( VecScatterCreateToAll(_srcNor, &ctx, &_srcAllNor) );
	 pC( VecScatterBegin(ctx, _srcNor, _srcAllNor, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterEnd(ctx,  _srcNor, _srcAllNor, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterDestroy(ctx) );
  }
  return(0);
}

// ----------------------------------------------------------------------
/* Dense3d_MPI evaluate function does a direct multiplication for a direct solution */
#undef __FUNCT__
#define __FUNCT__ "Dense3d_MPI::evaluate"
int Dense3d_MPI::evaluate(Vec srcDen, Vec trgVal) 
{
  //begin
  // CHECK
  pA(_srcPos!=NULL && _srcNor!=NULL && _trgPos!=NULL && srcDen!=NULL && trgVal!=NULL);
  //-----------------------------------
  int dim  = this->dim();
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  /* Get global number of source positions */
  PetscInt srcGlbNum = procGlbNum(_srcPos);
  /* Get local number of target positions */
  PetscInt trgLclNum = procLclNum(_trgPos);
  
  Vec srcAllPos = _srcAllPos;
  Vec srcAllNor = _srcAllNor;
  Vec srcAllDen;
  /* Create scatter context to scatter source densities to all processors */
  {
	 VecScatter ctx;
	 pC( VecScatterCreateToAll(srcDen, &ctx, &srcAllDen) );
	 pC( VecScatterBegin(ctx, srcDen, srcAllDen, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterEnd(ctx,  srcDen, srcAllDen, INSERT_VALUES, SCATTER_FORWARD) );
	 pC( VecScatterDestroy(ctx) );
  }
  
  Vec trgLclPos = _trgPos;
  Vec trgLclVal =  trgVal;

  /* Create matrices for source positions, normals, densities.  See common/nummat.hpp for
	* more information on matrices */
  double* srcAllPosArr; pC( VecGetArray(srcAllPos, &srcAllPosArr) );
  DblNumMat srcAllPosMat(dim, srcGlbNum, false, srcAllPosArr);
  double* srcAllNorArr; pC( VecGetArray(srcAllNor, &srcAllNorArr) );
  DblNumMat srcAllNorMat(dim, srcGlbNum, false, srcAllNorArr);
  double* srcAllDenArr; pC( VecGetArray(srcAllDen, &srcAllDenArr) );
  DblNumVec srcAllDenVec(srcDOF*srcGlbNum, false, srcAllDenArr);

  /* Create matrices for target positions and values */
  double* trgLclPosArr; pC( VecGetArray(trgLclPos, &trgLclPosArr) );
  DblNumMat trgLclPosMat(dim, trgLclNum, false, trgLclPosArr);
  double* trgLclValArr; pC( VecGetArray(trgLclVal, &trgLclValArr) );
  DblNumVec trgLclValVec(trgDOF*trgLclNum, false, trgLclValArr);

  /* Create an evaluation context and evaluate based on kernel type */
  DblNumMat inter(trgDOF, srcGlbNum*srcDOF);
  /* Do multiplication one line of the matrices at a time */
  for(int i=0; i<trgLclNum; i++) {
	 DblNumMat onePosMat(dim, 1, false, trgLclPosMat.clmdata(i));
	 DblNumVec oneValVec(trgDOF, false, trgLclValVec.data()+trgDOF*i);
	 /* Create kernel multiplier context based on kernel type */
	 pC( _knl.buildKnlIntCtx(srcAllPosMat, srcAllNorMat, onePosMat, inter) );
	 /* Computes 1.0*inter*srcAllDenVec + 0.0*oneValVec = oneValVec */
	 pC( dgemv(1.0, inter, srcAllDenVec, 0.0, oneValVec) );
  }
  
  pC( VecRestoreArray(srcAllPos, &srcAllPosArr) );
  pC( VecRestoreArray(srcAllNor, &srcAllNorArr) );
  pC( VecRestoreArray(srcAllDen, &srcAllDenArr) );
  
  pC( VecRestoreArray(trgLclPos, &trgLclPosArr) );
  pC( VecRestoreArray(trgLclVal, &trgLclValArr) );
  
  pC( VecDestroy(srcAllDen) );
  
  return(0);
}


