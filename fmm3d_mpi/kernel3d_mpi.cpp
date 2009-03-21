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
#include "common/vecmatop.hpp"
#include "kernel3d_mpi.hpp"
#include "comobject_mpi.hpp"

/** \file
 * This code implements the few Kernel3d_MPI functions.  For example,
 * based on kernelType, return source and target degrees of freedom.
 * Most importantly, based on the type of kernel being used
 * to do the evaluations, evaluate a multiplied context and the degrees of
 * freedom necessary in the buildKnlIntCtx function */


/* Most of this code is self-explanatory.  Based on the type of kernel being used
 * to do the evaluations, evaluate a multiplied context and the degrees of
 * freedom necessary */

/* Minimum difference */
double Kernel3d_MPI::_mindif = 1e-8;

// ----------------------------------------------------------------------
/* Depending on kernel type, set the degree of freedom for the sources
 * See kernel3d_mpi.hpp for kernelType descriptions */
int Kernel3d_MPI::srcDOF() const
{
  int dof = 0;
  switch(_kernelType) {
	 //laplace kernels
  case KNL_LAP_S_U: dof = 1; break;
  case KNL_LAP_D_U: dof = 1; break;
  case KNL_LAP_I  : dof = 1; break;
	 //stokes kernels
  case KNL_STK_F_U: dof = 4; break;
  case KNL_STK_S_U: dof = 3; break;
  case KNL_STK_S_P: dof = 3; break;
  case KNL_STK_D_U: dof = 3; break;
  case KNL_STK_D_P: dof = 3; break;
  case KNL_STK_R_U: dof = 3; break;
  case KNL_STK_R_P: dof = 3; break;
  case KNL_STK_I  : dof = 3; break;
  case KNL_STK_E  : dof = 3; break;
	 //navier kernels:	 //case KNL_NAV_F_U: dof = 3; break; //used for fmm
  case KNL_NAV_S_U: dof = 3; break;
  case KNL_NAV_D_U: dof = 3; break;
  case KNL_NAV_R_U: dof = 3; break;
  case KNL_NAV_I  : dof = 3; break;
  case KNL_NAV_E  : dof = 3; break;
	 //others
	 //error
  case KNL_ERR:     dof = 0; break;
  }
  return dof;
}

// ----------------------------------------------------------------------
/* Depending on the kernel type, set the degree of freedom for the target values
 * See kernel3d_mpi.hpp for kernelType descriptions */
int Kernel3d_MPI::trgDOF() const
{
  int dof = 0;
  switch(_kernelType) {
	 //laplace kernels
  case KNL_LAP_S_U: dof = 1; break;
  case KNL_LAP_D_U: dof = 1; break;
  case KNL_LAP_I  : dof = 1; break;
	 //stokes kernels
  case KNL_STK_F_U: dof = 3; break;
  case KNL_STK_S_U: dof = 3; break;
  case KNL_STK_S_P: dof = 1; break;
  case KNL_STK_D_U: dof = 3; break;
  case KNL_STK_D_P: dof = 1; break;
  case KNL_STK_R_U: dof = 3; break;
  case KNL_STK_R_P: dof = 1; break;
  case KNL_STK_I  : dof = 3; break;
  case KNL_STK_E  : dof = 3; break;
	 //navier kernels:	 //  case KNL_NAV_F_U: dof = 3; break; //used for fmm
  case KNL_NAV_S_U: dof = 3; break;
  case KNL_NAV_D_U: dof = 3; break;
  case KNL_NAV_R_U: dof = 3; break;
  case KNL_NAV_I  : dof = 3; break;
  case KNL_NAV_E  : dof = 3; break;
	 //others
	 //error
  case KNL_ERR:     dof = 0; break;
  }
  return dof;
}

// ---------------------------------------------------------------------- 
bool Kernel3d_MPI::homogeneous() const
{
  bool ret = false;
  switch(_kernelType) {
	 //laplace kernels
  case KNL_LAP_S_U: ret = true; break;
	 //stokes kernels
  case KNL_STK_F_U: ret = true; break;
	 //stokes kernels
  case KNL_NAV_S_U: ret = true; break;
  default: assert(0);
  }
  return ret;
}

// ---------------------------------------------------------------------- 
void Kernel3d_MPI::homogeneousDeg(vector<double>& degVec) const
{
  switch(_kernelType) {
  case KNL_LAP_S_U: degVec.resize(1); degVec[0]=1; break;
  case KNL_STK_F_U: degVec.resize(4); degVec[0]=1; degVec[1]=1; degVec[2]=1; degVec[3]=2; break;
  case KNL_NAV_S_U: degVec.resize(3); degVec[0]=1; degVec[1]=1; degVec[2]=1; break;
  default: assert(0);
  }
  return;
}

// ----------------------------------------------------------------------
/* This is the main function for Kernel3d_MPI.
 * Given source positions, normals and target positions, build a matrix inter
 * which will be used as a multiplier during the multiplication phase.
 * This functino is just a set of cases, based on the kernelType variable.
 * Mathematical details on these kernels are available in the papers
 */
#undef __FUNCT__
#define __FUNCT__ "Kernel3d_MPI::kernel"
int Kernel3d_MPI::buildKnlIntCtx(const DblNumMat& srcPos, const DblNumMat& srcNor, const DblNumMat& trgPos, DblNumMat& inter)
{
  //begin
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  pA(srcPos.m()==dim() && srcNor.m()==dim() && trgPos.m()==dim());
  pA(srcPos.n()==srcNor.n());
  pA(srcPos.n()*srcDOF == inter.n());
  pA(trgPos.n()*trgDOF == inter.m());
  /* Single-Layer Laplacian Position/Velocity */
  if(       _kernelType==KNL_LAP_S_U) {
	 //----------------------------------------------------------------------------------
	 //---------------------------------
	 double OOFP = 1.0/(4.0*M_PI);
	 for(int i=0; i<trgPos.n(); i++) {
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  				  double r = sqrt(r2);
		  if(r<_mindif) {
			 inter(i,j) = 0;
		  } else {
			 inter(i,j) = OOFP / r;
		  }
		}
	 }
  }
  /* Double-Layer Laplacian Position/Velocity */
  else if(_kernelType==KNL_LAP_D_U) {
	 //---------------------------------
	 double OOFP = 1.0/(4.0*M_PI);
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
			 double rn = x*nx + y*ny + z*nz;
			 double r3 = r2*r;
			 inter(i,j) = - OOFP / r3 * rn;
		  }
		}
  }
  /* Laplacian Identity Kernel */
  else if(_kernelType==KNL_LAP_I) {
	 //---------------------------------
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++)
		  inter(i,j) = 1;
  }
  /* Stokes - Fmm3d velocity kernel */
  else if(_kernelType==KNL_STK_F_U) {
	 //----------------------------------------------------------------------------------
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 double mu = _coefs[0];
	 double OOFP = 1.0/(4.0*M_PI);
	 double OOEP = 1.0/(8.0*M_PI);
	 double oomu = 1.0/mu;
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 double G = oomu * OOEP / r;
			 double H = oomu * OOEP / r3;
			 double A = OOFP / r3;
			 int is = i*3;			 int js = j*4;
			 inter(is,   js)   = G + H*x*x; inter(is,   js+1) =     H*x*y; inter(is,   js+2) =     H*x*z; inter(is,   js+3) = A*x;
			 inter(is+1, js)   =     H*y*x; inter(is+1, js+1) = G + H*y*y; inter(is+1, js+2) =     H*y*z; inter(is+1, js+3) = A*y;
			 inter(is+2, js)   =     H*z*x; inter(is+2, js+1) =     H*z*y; inter(is+2, js+2) = G + H*z*z; inter(is+2, js+3) = A*z;
		  }
		}
  }
  /* Stokes Single-Layer Position/Velocity kernel */
  else if(_kernelType==KNL_STK_S_U) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 double mu = _coefs[0];
	 double OOEP = 1.0/(8.0*M_PI);
	 double oomu = 1.0/mu;
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 double G = OOEP / r;
			 double H = OOEP / r3;
			 inter(i*3,   j*3)   = oomu*(G + H*x*x); inter(i*3,   j*3+1) = oomu*(    H*x*y); inter(i*3,   j*3+2) = oomu*(    H*x*z);
			 inter(i*3+1, j*3)   = oomu*(    H*y*x); inter(i*3+1, j*3+1) = oomu*(G + H*y*y); inter(i*3+1, j*3+2) = oomu*(    H*y*z);
			 inter(i*3+2, j*3)   = oomu*(    H*z*x); inter(i*3+2, j*3+1) = oomu*(    H*z*y); inter(i*3+2, j*3+2) = oomu*(G + H*z*z);
		  }
		}
  }
  /* Stokes Single-Layer Pressure kernel */
  else if(_kernelType==KNL_STK_S_P) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 // double mu = _coefs[0];
	 double OOFP = 1.0/(4.0*M_PI);
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 inter(i  ,j*3  ) = OOFP*x/r3;			 inter(i  ,j*3+1) = OOFP*y/r3;			 inter(i  ,j*3+2) = OOFP*z/r3;
		  }
		}
  }
  /* Stokes Double-Layer Position/Velocity kernel */
  else if(_kernelType==KNL_STK_D_U) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 // double mu = _coefs[0];
	 double SOEP = 6.0/(8.0*M_PI);
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
			 double rn = x*nx + y*ny + z*nz;
			 double r5 = r2*r2*r;
			 double C = - SOEP / r5;
			 inter(i*3,   j*3)   = C*rn*x*x;				  inter(i*3,   j*3+1) = C*rn*x*y;				  inter(i*3,   j*3+2) = C*rn*x*z;
			 inter(i*3+1, j*3)   = C*rn*y*x;				  inter(i*3+1, j*3+1) = C*rn*y*y;				  inter(i*3+1, j*3+2) = C*rn*y*z;
			 inter(i*3+2, j*3)   = C*rn*z*x;				  inter(i*3+2, j*3+1) = C*rn*z*y;				  inter(i*3+2, j*3+2) = C*rn*z*z;
		  }
		}
  }
  /* Stokes Double-Layer Pressure kernel */
  else if(_kernelType==KNL_STK_D_P) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 double mu = _coefs[0];
	 double OOTP = 1.0/(2.0*M_PI);
	 double coef = mu*OOTP;
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
			 double rn = x*nx + y*ny + z*nz;
			 double r3 = r2*r;
			 double r5 = r3*r2;
			 int is = i;			 int js = j*3;
			 inter(is  ,js  ) = coef*(nx/r3 - 3*rn*x/r5);
			 inter(is  ,js+1) = coef*(ny/r3 - 3*rn*y/r5);
			 inter(is  ,js+2) = coef*(nz/r3 - 3*rn*z/r5);
		  }
		}
  }
  else if(_kernelType==KNL_STK_R_U) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 double mu = _coefs[0];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 double coef = 1.0/(8.0*M_PI*mu)/r3;
			 inter(i*3,   j*3)   = coef*0;			 inter(i*3,   j*3+1) = coef*z;			 inter(i*3,   j*3+2) = coef*(-y);
			 inter(i*3+1, j*3)   = coef*(-z);		 inter(i*3+1, j*3+1) = coef*0;			 inter(i*3+1, j*3+2) = coef*x;
			 inter(i*3+2, j*3)   = coef*y;			 inter(i*3+2, j*3+1) = coef*(-x);		 inter(i*3+2, j*3+2) = coef*0;
		  }
		}
  } else if(_kernelType==KNL_STK_R_P) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 // double mu = _coefs[0];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  inter(i,j*3  ) = 0;
		  inter(i,j*3+1) = 0;
		  inter(i,j*3+2) = 0;
		}
  } else if(_kernelType==KNL_STK_I) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 // double mu = _coefs[0];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  inter(i*3,   j*3  ) = 1;		 inter(i*3,   j*3+1) = 0;		 inter(i*3,   j*3+2) = 0;
		  inter(i*3+1, j*3  ) = 0;		 inter(i*3+1, j*3+1) = 1;		 inter(i*3+1, j*3+2) = 0;
		  inter(i*3+2, j*3  ) = 0;		 inter(i*3+2, j*3+1) = 0;		 inter(i*3+2, j*3+2) = 1;
		}
  }
  /* levi-civita tensor */
  else if(_kernelType==KNL_STK_E) {
	 //---------------------------------
	 pA(_coefs.size()>=1);
	 // double mu = _coefs[0];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  inter(i*3,   j*3  ) = 0;		 inter(i*3,   j*3+1) = z;		 inter(i*3,   j*3+2) = -y;
		  inter(i*3+1, j*3  ) = -z;  	 inter(i*3+1, j*3+1) = 0;		 inter(i*3+1, j*3+2) = x;
		  inter(i*3+2, j*3  ) = y;		 inter(i*3+2, j*3+1) = -x;		 inter(i*3+2, j*3+2) = 0;
		}
  }
  /* Navier-Stokes Single-Layer Position/Velocity Kernel */
  else if(_kernelType==KNL_NAV_S_U) {
	 //----------------------------------------------------------------------------------
	 //---------------------------------
	 pA(_coefs.size()>=2);
	 double mu = _coefs[0];	 double ve = _coefs[1];
	 double sc1 = (3.0-4.0*ve)/(16.0*M_PI*(1.0-ve));
	 double sc2 = 1.0/(16.0*M_PI*(1.0-ve));
	 double oomu = 1.0/mu;
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 double G = sc1 / r;
			 double H = sc2 / r3;
			 inter(i*3,   j*3)   = oomu*(G + H*x*x);  inter(i*3,   j*3+1) = oomu*(    H*x*y);  inter(i*3,   j*3+2) = oomu*(    H*x*z);
			 inter(i*3+1, j*3)   = oomu*(    H*y*x);  inter(i*3+1, j*3+1) = oomu*(G + H*y*y);  inter(i*3+1, j*3+2) = oomu*(    H*y*z);
			 inter(i*3+2, j*3)   = oomu*(    H*z*x);  inter(i*3+2, j*3+1) = oomu*(    H*z*y);  inter(i*3+2, j*3+2) = oomu*(G + H*z*z);
		  }
		}
  }
  /* Navier-Stokes Double-Layer Position/Velocity Kernel */
  else if(_kernelType==KNL_NAV_D_U) {
	 //---------------------------------
	 pA(_coefs.size()>=2);
	 // double mu = _coefs[0];	 
	 double ve = _coefs[1];
	 double dc1 = -(1-2.0*ve)/(8.0*M_PI*(1.0-ve));
	 double dc2 =  (1-2.0*ve)/(8.0*M_PI*(1.0-ve));
	 double dc3 = -3.0/(8.0*M_PI*(1.0-ve));
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double nx = srcNor(0,j);				  double ny = srcNor(1,j);				  double nz = srcNor(2,j);
			 double rn = x*nx + y*ny + z*nz;
			 double r3 = r2*r;
			 double r5 = r3*r2;
			 double A = dc1 / r3;			 double B = dc2 / r3;			 double C = dc3 / r5;
			 double rx = x;			 double ry = y;			 double rz = z;
			 //&&&&&&&
			 inter(i*3,   j*3)   = A*(rn+nx*rx) + B*(rx*nx) + C*rn*rx*rx;
			 inter(i*3,   j*3+1) = A*(   nx*ry) + B*(rx*ny) + C*rn*rx*ry;
			 inter(i*3,   j*3+2) = A*(   nx*rz) + B*(rx*nz) + C*rn*rx*rz;
			 inter(i*3+1, j*3)   = A*(   ny*rx) + B*(ry*nx) + C*rn*ry*rx;
			 inter(i*3+1, j*3+1) = A*(rn+ny*ry) + B*(ry*ny) + C*rn*ry*ry;
			 inter(i*3+1, j*3+2) = A*(   ny*rz) + B*(ry*nz) + C*rn*ry*rz;
			 inter(i*3+2, j*3)   = A*(   nz*rx) + B*(rz*nx) + C*rn*rz*rx;
			 inter(i*3+2, j*3+1) = A*(   nz*ry) + B*(rz*ny) + C*rn*rz*ry;
			 inter(i*3+2, j*3+2) = A*(rn+nz*rz) + B*(rz*nz) + C*rn*rz*rz;
		  }
		}
  } else if(_kernelType==KNL_NAV_R_U) {
	 //---------------------------------
	 pA(_coefs.size()>=2);
	 double mu = _coefs[0];	 // double ve = _coefs[1];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  double r2 = x*x + y*y + z*z;			  double r = sqrt(r2);
		  if(r<_mindif) {
			 for(int t=0;t<trgDOF;t++) for(int s=0;s<srcDOF;s++) { inter(i*trgDOF+t, j*srcDOF+s) = 0.0; }
		  } else {
			 double r3 = r2*r;
			 double coef = 1.0/(8.0*M_PI*mu)/r3;
			 inter(i*3,   j*3)   = coef*0;			 inter(i*3,   j*3+1) = coef*z;			 inter(i*3,   j*3+2) = coef*(-y);
			 inter(i*3+1, j*3)   = coef*(-z);		 inter(i*3+1, j*3+1) = coef*0;			 inter(i*3+1, j*3+2) = coef*x;
			 inter(i*3+2, j*3)   = coef*y;			 inter(i*3+2, j*3+1) = coef*(-x);		 inter(i*3+2, j*3+2) = coef*0;
		  }
		}
  } else if(_kernelType==KNL_NAV_I) {
	 //---------------------------------
	 pA(_coefs.size()>=2);
	 // double mu = _coefs[0];	 
	 // double ve = _coefs[1];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  inter(i*3,   j*3  ) = 1;		 inter(i*3,   j*3+1) = 0;		 inter(i*3,   j*3+2) = 0;
		  inter(i*3+1, j*3  ) = 0;		 inter(i*3+1, j*3+1) = 1;		 inter(i*3+1, j*3+2) = 0;
		  inter(i*3+2, j*3  ) = 0;		 inter(i*3+2, j*3+1) = 0;		 inter(i*3+2, j*3+2) = 1;
		}
  } else if(_kernelType==KNL_NAV_E) {
	 //---------------------------------
	 pA(_coefs.size()>=2);
	 // double mu = _coefs[0];	 
	 // double ve = _coefs[1];
	 for(int i=0; i<trgPos.n(); i++)
		for(int j=0; j<srcPos.n(); j++) {
		  double x = trgPos(0,i) - srcPos(0,j); double y = trgPos(1,i) - srcPos(1,j); double z = trgPos(2,i) - srcPos(2,j);
		  inter(i*3,   j*3  ) = 0;		 inter(i*3,   j*3+1) = z;		 inter(i*3,   j*3+2) = -y;
		  inter(i*3+1, j*3  ) = -z;  	 inter(i*3+1, j*3+1) = 0;		 inter(i*3+1, j*3+2) = x;
		  inter(i*3+2, j*3  ) = y;		 inter(i*3+2, j*3+1) = -x;		 inter(i*3+2, j*3+2) = 0;
		}
  } else if(_kernelType==KNL_ERR) {
	 //---------------------------------
	 pA(0);
  }
  return(0);
}




