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
#include "matmgnt3d_mpi.hpp"
#include "common/vecmatop.hpp"

using std::cerr;
using std::endl;

// ---------------------------------------------------------------------- 
double MatMgnt3d_MPI::_wsbuf[16384];

// ---------------------------------------------------------------------- 
MatMgnt3d_MPI::MatMgnt3d_MPI(): _np(6)
{
}
// ---------------------------------------------------------------------- 
MatMgnt3d_MPI::~MatMgnt3d_MPI()
{
}

/* alt() returns 0.1^(_np+1).  This is used as a tolerance level in computing the pesudoinverse, pinv, below */
double MatMgnt3d_MPI::alt()
{
  return pow(0.1, _np+1);
}
// ---------------------------------------------------------------------- 
// ----------------------------------------------------------------------
/* Setup the MatMgnt3d_MPI based on the kernel.
 * Also, calculate the sample positions for upward equivalent,
 * upward-check, downward-equivalent and downward-check surfaces.
 * These are basic matrices, whose sizes are determined by the level of
 * precision, _np, and stored in _samPos[enum].
 * Also, calculate regular positions, _regPos
 * This functino is called by MatMgnt3d_MPI* MatMgnt3d_MPI::getmmptr(Kernel3d_MPI knl, int np)
 * during the FMM3d_MPI::setup() phase.
 */
int MatMgnt3d_MPI::setup()
{
  //--------------------------------------------------------
  _homogeneous = _knl.homogeneous();
  if(_homogeneous==true) {
	 _knl.homogeneousDeg(_degVec); pA(_degVec.size()==(unsigned)srcDOF());
  }
  /* Calculate sample positions for different surfaces.  For different sized nodes,
	* these positions will be scaled and shifted to local positions during upward/downward
	* multiplication (see fmm3d_mpi.cpp) using the locPos function below
	*/
  pC( samPosCal(_np,   1.0, _samPos[UE]) );
  pC( samPosCal(_np+2, 3.0, _samPos[UC]) );
  pC( samPosCal(_np,   3.0, _samPos[DE]) );
  pC( samPosCal(_np,   1.0, _samPos[DC]) );
  
  pC( regPosCal(_np,   1.0, _regPos    ) ); //only one regPos
  //--------------------------------------------------------
  return (0);
}
// ----------------------------------------------------------------------
/* Report the sizes of the matrix maps */
int MatMgnt3d_MPI::report()
{
  cerr<<"matrix map size"<<endl;
  cerr<<_upwChk2UpwEqu.size()<<" "<<_upwEqu2UpwChk.size()<<" "<<_dwnChk2DwnEqu.size()<<" "<<_dwnEqu2DwnChk.size()<<" "<<_upwEqu2DwnChk.size()<<endl;
  return (0);
}

// ----------------------------------------------------------------------
/* Size of the plain data for the different surfaces (UE, UC, DE, DC) */
int MatMgnt3d_MPI::plnDatSze(int tp)
{
  if(tp==UE || tp==DE)
	 return _samPos[tp].n()*srcDOF();
  else
	 return _samPos[tp].n()*trgDOF();
}

// ----------------------------------------------------------------------
/* Size of the effective data for the different surfaces (UE, UC, DE, DC)
 * This is for use with FFT acceleration when plain data is converted to
 * "effective data" in int plnDen2EffDen(int level, const DblNumVec&, DblNumVec&)
 * below
 */
int MatMgnt3d_MPI::effDatSze(int tp)
{
  int effnum = (2*_np+2)*(2*_np)*(2*_np);
  if(tp==UE || tp==DE)
	 return effnum*srcDOF();
  else
	 return effnum*trgDOF();
}

// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::UpwChk2UpwEqu_dgemv(int l, const DblNumVec& chk, DblNumVec& den)
{
  DblNumMat& _UpwChk2UpwEqu = (_homogeneous==true) ? _upwChk2UpwEqu[0] : _upwChk2UpwEqu[l];
  double R = (_homogeneous==true) ? 1 : 1.0/pow(2.0,l);
  //---------compute matrix
  if(_UpwChk2UpwEqu.m()==0) {	
	 //set matrix
	 DblNumMat ud2c(plnDatSze(UC), plnDatSze(UE));
	 DblNumMat chkPos(dim(),samPos(UC).n());
	 clear(chkPos);
	 pC( daxpy(R, samPos(UC), chkPos) ); //scale

	 DblNumMat denPos(dim(),samPos(UE).n());
	 clear(denPos);
	 pC( daxpy(R, samPos(UE), denPos) ); //scale
	 
	 pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, ud2c) );
	 _UpwChk2UpwEqu.resize(plnDatSze(UE), plnDatSze(UC));
	 /* pseudoinverse.  See common/vecmatop.hpp, common/vecmatop.cpp */
	 pC( pinv(ud2c, alt(), _UpwChk2UpwEqu) );
  }
  //---------matvec
  if(_homogeneous==true) {
	 //matvec
	 int srcDOF = this->srcDOF();
	 DblNumVec tmpDen(srcDOF*samPos(UE).n(), false, _wsbuf);	 clear(tmpDen);
	 pC( dgemv(1.0, _UpwChk2UpwEqu, chk, 1.0, tmpDen) );
	 //scale
	 vector<double> sclvec(srcDOF);
	 for(int s=0; s<srcDOF; s++)
		sclvec[s] = pow(2.0, - l*_degVec[s]);
	 int cnt = 0;
	 for(int i=0; i<samPos(UE).n(); i++) {
		for(int s=0; s<srcDOF; s++) {
		  den(cnt) = den(cnt) + tmpDen(cnt) * sclvec[s];
		  cnt++;
		}
	 }
  } else {
	 pC( dgemv(1.0, _UpwChk2UpwEqu, chk, 1.0, den) );
  }
  return (0);
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::UpwEqu2UpwChk_dgemv(int l, Index3 idx, const DblNumVec& den, DblNumVec& chk)
{
  NumTns<DblNumMat>& _UpwEqu2UpwChk = (_homogeneous==true) ? _upwEqu2UpwChk[0] : _upwEqu2UpwChk[l];
  double R = (_homogeneous==true) ? 1 : 1.0/pow(2.0, l);  
  if(_UpwEqu2UpwChk.m()==0)	 _UpwEqu2UpwChk.resize(2,2,2);
  DblNumMat& _UpwEqu2UpwChkii = _UpwEqu2UpwChk(idx(0), idx(1), idx(2));
  //---------compute matrix
  if(_UpwEqu2UpwChkii.m()==0) {	 
	 _UpwEqu2UpwChkii.resize(plnDatSze(UC), plnDatSze(UE)); 
	 DblNumMat chkPos(dim(),samPos(UC).n());
	 clear(chkPos);
	 pC( daxpy(2.0*R, samPos(UC), chkPos) ); //scale

	 DblNumMat denPos(dim(),samPos(UE).n());
	 clear(denPos);
	 pC( daxpy(R, samPos(UE), denPos) ); //scale

	 for(int i=0; i<dim(); i++) {
		for(int j=0; j<samPos(UE).n(); j++)	{
		  denPos(i,j) = denPos(i,j) + (2*idx(i)-1)*R;//shift
		}
	 }
	 pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, _UpwEqu2UpwChkii) );
  }
  //---------matvec
  if(_homogeneous==true) {
	 int srcDOF = this->srcDOF();
	 DblNumVec tmpDen(srcDOF*samPos(UE).n(), false, _wsbuf);	 clear(tmpDen);
	 vector<double> sclvec(srcDOF);
	 for(int s=0; s<srcDOF; s++)
		sclvec[s] = pow(2.0, l*_degVec[s]);
	 int cnt = 0;
	 for(int i=0; i<samPos(UE).n(); i++) {
		for(int s=0; s<srcDOF; s++) {
		  tmpDen(cnt) = den(cnt) * sclvec[s];		  cnt++;
		}
	 }
	 pC( dgemv(1.0, _UpwEqu2UpwChkii, tmpDen, 1.0, chk) );
  } else {
	 pC( dgemv(1.0, _UpwEqu2UpwChkii, den, 1.0, chk) );
  }
  return (0);
}

// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::DwnChk2DwnEqu_dgemv(int l, const DblNumVec& chk, DblNumVec& den)
{
  DblNumMat& _DwnEqu2DwnChk = (_homogeneous==true) ? _dwnChk2DwnEqu[0]: _dwnChk2DwnEqu[l];
  double R = (_homogeneous==true) ? 1 : 1.0/pow(2.0,l);
  //---------compute matrix
  if(_DwnEqu2DwnChk.m()==0) {	 
	 DblNumMat dd2c(plnDatSze(DC), plnDatSze(DE));
	 DblNumMat chkPos(dim(),samPos(DC).n());		clear(chkPos);	 pC( daxpy(R, samPos(DC), chkPos) ); //scale
	 DblNumMat denPos(dim(),samPos(DE).n());		clear(denPos);	 pC( daxpy(R, samPos(DE), denPos) ); //scale
	 
	 pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, dd2c) );//matrix
	 _DwnEqu2DwnChk.resize(plnDatSze(DE), plnDatSze(DC));
	 /* psuedoinverse */
	 pC( pinv(dd2c, alt(), _DwnEqu2DwnChk) );
  }
  //---------matvec
  if(_homogeneous==true) {
	 int srcDOF = this->srcDOF();
	 DblNumVec tmpDen(srcDOF*samPos(DE).n(), false, _wsbuf);	 clear(tmpDen);
	 pC( dgemv(1.0, _DwnEqu2DwnChk, chk, 1.0, tmpDen) );
	 //scale
	 vector<double> sclvec(srcDOF);
	 for(int s=0; s<srcDOF; s++)
		sclvec[s] = pow(2.0, - l*_degVec[s]);
	 int cnt = 0;
	 for(int i=0; i<samPos(DE).n(); i++)
		for(int s=0; s<srcDOF; s++) {
		  den(cnt) = den(cnt) + tmpDen(cnt) * sclvec[s];		  cnt++;
		}
  } else {
	 pC( dgemv(1.0, _DwnEqu2DwnChk, chk, 1.0, den) );
  }
  return 0;
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::DwnEqu2DwnChk_dgemv(int l, Index3 idx, const DblNumVec& den, DblNumVec& chk)
{
  NumTns<DblNumMat>& _DwnEqu2DwnChk = (_homogeneous==true) ? _dwnEqu2DwnChk[0] : _dwnEqu2DwnChk[l];
  double R = (_homogeneous==true) ? 1 : 1.0/pow(2.0, l);  
  if(_DwnEqu2DwnChk.m()==0)	 _DwnEqu2DwnChk.resize(2,2,2);
  DblNumMat& _DwnEqu2DwnChkii = _DwnEqu2DwnChk(idx[0], idx[1], idx[2]);
  
  //---------compute matrix
  if(_DwnEqu2DwnChkii.m()==0) {	 
	 _DwnEqu2DwnChkii.resize(plnDatSze(DC), plnDatSze(DE)); 
	 DblNumMat denPos(dim(),samPos(DE).n());
	 clear(denPos);
	 pC( daxpy(R, samPos(DE), denPos) ); //scale

	 DblNumMat chkPos(dim(),samPos(DC).n());
	 clear(chkPos);
	 pC( daxpy(0.5*R, samPos(DC), chkPos) ); //scale

	 for(int i=0; i<dim(); i++) {
		for(int j=0; j<samPos(DC).n(); j++) {
		  chkPos(i,j) = chkPos(i,j) + (double(idx(i))-0.5)*R;
		}
	 }
	 pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, _DwnEqu2DwnChkii) );
  }
  //---------matvec
  if(_homogeneous==true) {
	 int srcDOF = this->srcDOF();
	 DblNumVec tmpDen(srcDOF*samPos(DE).n(), false, _wsbuf);	 clear(tmpDen);
	 vector<double> sclvec(srcDOF);
	 for(int s=0; s<srcDOF; s++)
		sclvec[s] = pow(2.0, l*_degVec[s]);
	 int cnt = 0;
	 for(int i=0; i<samPos(DE).n(); i++) {
		for(int s=0; s<srcDOF; s++) {
		  tmpDen(cnt) = den(cnt) * sclvec[s];		  cnt++;
		}
	 }
	 pC( dgemv(1.0, _DwnEqu2DwnChkii, tmpDen, 1.0, chk) );
  } else {
	 pC( dgemv(1.0, _DwnEqu2DwnChkii, den, 1.0, chk) );
  }
  return (0);
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::plnDen2EffDen(int l, const DblNumVec& plnDen, DblNumVec& effDen)
{
  DblNumVec regDen(regPos().n()*srcDOF());
  clear(regDen);
  if(_homogeneous==true) {
	 int srcDOF = this->srcDOF();
	 DblNumVec tmpDen(srcDOF*samPos(UE).n(), false, _wsbuf);	 clear(tmpDen);
	 vector<double> sclvec(srcDOF);
	 for(int s=0; s<srcDOF; s++)
		sclvec[s] = pow(2.0, l*_degVec[s]);
	 int cnt = 0;
	 for(int i=0; i<samPos(UE).n(); i++) {
		for(int s=0; s<srcDOF; s++) {
		  tmpDen(cnt) = plnDen(cnt) * sclvec[s];		  cnt++;
		}
	 }
	 pC( samDen2RegDen(tmpDen, regDen) );
  } else {
	 pC( samDen2RegDen(plnDen, regDen) );
  }
  
  int nnn[3];
  nnn[0] = 2*_np;
  nnn[1] = 2*_np;
  nnn[2] = 2*_np;

  fftw_plan forplan = fftw_plan_many_dft_r2c(3,nnn,srcDOF(), regDen.data(),NULL, srcDOF(),1, (fftw_complex*)(effDen.data()),NULL, srcDOF(),1, FFTW_ESTIMATE);
  fftw_execute(forplan);
  fftw_destroy_plan(forplan);  
  
  return (0);
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::samDen2RegDen(const DblNumVec& samDen, DblNumVec& regDen)
{
  int np = _np;
  int rgnum = 2*np;
  int srcDOF = this->srcDOF();
  int count=0;
  //the order of iterating is the same as SampleGrid
  for(int i=0; i<np; i++) {
	 for(int j=0; j<np; j++) {
		for(int k=0; k<np; k++) {
		  if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
			 //the position is fortran style
			 int rgoff = (k+np/2)*rgnum*rgnum + (j+np/2)*rgnum + (i+np/2);
			 for(int f=0; f<srcDOF; f++) {
				regDen(srcDOF*rgoff + f) += samDen(srcDOF*count + f);
			 }	
			 count++;
		  }
		}
	 }
  }
  return 0;
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::effVal2PlnVal(int level, const DblNumVec& effVal, DblNumVec& plnVal)
{
  DblNumVec regVal(regPos().n()*trgDOF());
  int nnn[3];
  nnn[0] = 2*_np;
  nnn[1] = 2*_np;
  nnn[2] = 2*_np;
  
  fftw_plan invplan = fftw_plan_many_dft_c2r(3,nnn,trgDOF(), (fftw_complex*)(effVal.data()),NULL, trgDOF(),1, regVal.data(),NULL, trgDOF(),1, FFTW_ESTIMATE);
  fftw_execute(invplan);
  fftw_destroy_plan(invplan); 
  
  pC( regVal2SamVal(regVal, plnVal) );
  return (0);
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::regVal2SamVal(const DblNumVec& regVal, DblNumVec& samVal)
{
  int np = _np;
  int rgnum = 2*np;
  int trgDOF = this->trgDOF();
  int count=0;
  //the order of iterating is the same as SampleGrid
  for(int i=0; i<np; i++) {
	 for(int j=0; j<np; j++) {
		for(int k=0; k<np; k++) {
		  if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
			 //the position is fortran style
			 int rgoff = (k+np/2)*rgnum*rgnum + (j+np/2)*rgnum + (i+np/2);
			 for(int f=0; f<trgDOF; f++) {
				samVal(trgDOF*count + f) += regVal(trgDOF*rgoff + f);
			 }
			 count++;
		  }
		}
	 }
  }
  return 0;
}
// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::UpwEqu2DwnChk_dgemv(int l, Index3 idx, const DblNumVec& effDen, DblNumVec& effVal)
{
  OffTns<DblNumMat>& _UpwEqu2DwnChk = (_homogeneous==true) ? _upwEqu2DwnChk[0] : _upwEqu2DwnChk[l];
  double R = (_homogeneous==true) ? 1.0 : 1.0/pow(2.0, l); 
  if(_UpwEqu2DwnChk.m()==0)	 _UpwEqu2DwnChk.resize(7,7,7,-3,-3,-3);
  DblNumMat& _UpwEqu2DwnChkii = _UpwEqu2DwnChk(idx[0], idx[1], idx[2]);
  
  int effnum = (2*_np+2)*(2*_np)*(2*_np);
  int srcDOF = this->srcDOF();
  int trgDOF = this->trgDOF();
  
  //---------compute matrix
  if(_UpwEqu2DwnChkii.m()==0) { 
	 //-----------------------	 
	 pA( idx.linfty()>1 );
	 DblNumMat denPos(dim(),1);	 for(int i=0; i<dim(); i++)		denPos(i,0) = double(idx(i))*2.0*R; //shift
	 DblNumMat chkPos(dim(),regPos().n());	 clear(chkPos);	 pC( daxpy(R, regPos(), chkPos) );
	 DblNumMat tt(regPos().n()*trgDOF, srcDOF);
	 pC( _knl.buildKnlIntCtx(denPos, denPos, chkPos, tt) );
	 // move data to tmp
	 DblNumMat tmp(trgDOF,regPos().n()*srcDOF);
	 for(int k=0; k<regPos().n();k++) {
		for(int i=0; i<trgDOF; i++)
		  for(int j=0; j<srcDOF; j++) {
			 tmp(i,j+k*srcDOF) = tt(i+k*trgDOF,j);
		  }
	 }
	 _UpwEqu2DwnChkii.resize(trgDOF*srcDOF, effnum); 
	 //forward FFT from tmp to _UpwEqu2DwnChkii;
	 
	 int nnn[3];
	 nnn[0] = 2*_np;
	 nnn[1] = 2*_np;
	 nnn[2] = 2*_np;

	 fftw_plan forplan = fftw_plan_many_dft_r2c(3,nnn,srcDOF*trgDOF, tmp.data(),NULL, srcDOF*trgDOF,1, (fftw_complex*)(_UpwEqu2DwnChkii.data()),NULL, srcDOF*trgDOF,1, FFTW_ESTIMATE);
	 fftw_execute(forplan);
	 fftw_destroy_plan(forplan);	 
  }
  //---------matvec
  //fft mult
  double nrmfc = 1.0/double(regPos().n());
  fftw_complex* matptr = (fftw_complex*)(_UpwEqu2DwnChkii.data());
  fftw_complex* denptr = (fftw_complex*)(effDen.data());
  fftw_complex* chkptr = (fftw_complex*)(effVal.data());
  int matstp = srcDOF*trgDOF;
  int denstp = srcDOF;
  int chkstp = trgDOF;
  
  double newalpha = nrmfc;
  for(int i=0; i<trgDOF; i++)
	 for(int j=0; j<srcDOF; j++) {
		int matoff = j*trgDOF + i;
		int denoff = j;
		int chkoff = i;
		pC( cptwvv(effnum/2, newalpha, matptr+matoff, matstp, denptr+denoff, denstp, chkptr+chkoff, chkstp) );
	 }
  return (0);
}

// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::locPos(int tp, Point3 ctr, double rad, DblNumMat& pos)
{
  /* Get sample positions associated with UC, UE, DC, or DE */
  const DblNumMat& bas = samPos(tp);
  pos.resize(dim(), bas.n());
  /* Build pos matrix, which will hold the sample positions as
	* desired by the "tp" argument, but these sample positions
	* will be translated by the center ctr and radius rad of the target */
  for(int i=0; i<dim(); i++) {
	 for(int j=0; j<pos.n(); j++) {
		pos(i,j) = ctr(i) + rad * bas(i,j);
	 }
  }
  return (0);
}

// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::samPosCal(int np, double R, DblNumMat& pos)
{
  int n = np*np*np - (np-2)*(np-2)*(np-2);
  pos.resize(dim(),n);
  double step = 2.0/(np-1);
  double init = -1.0;
  int count = 0;
  for(int i=0; i<np; i++) {
	 for(int j=0; j<np; j++) {
		for(int k=0; k<np; k++) {
		  if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
			 double x = init + i*step;
			 double y = init + j*step;
			 double z = init + k*step;
			 pos(0,count) = R*x;
			 pos(1,count) = R*y;
			 pos(2,count) = R*z;
			 count++;
		  }
		}	
	 }	
  }
  pA(count==n);
  return 0;
}

// ---------------------------------------------------------------------- 
int MatMgnt3d_MPI::regPosCal(int np, double R, DblNumMat& pos)
{
  int n = 2*np*2*np*2*np;
  pos.resize(dim(), n);
  double step = 2.0/(np-1);
  int count = 0;
  for(int k=0; k<2*np; k++) {
	 for(int j=0; j<2*np; j++) {
		for(int i=0; i<2*np; i++) {
		  int gi = (i<np) ? i : i-2*np;
		  int gj = (j<np) ? j : j-2*np;
		  int gk = (k<np) ? k : k-2*np;
		  pos(0, count) = R * gi*step;
		  pos(1, count) = R * gj*step;
		  pos(2, count) = R * gk*step;
		  count ++;
		}
	 }
  }
  pA(count==n);
  return 0;
}

// ---------------------------------------------------------------------- 
inline int MatMgnt3d_MPI::cptwvv(int n, double alpha, fftw_complex* x, int incx, fftw_complex* y, int incy, fftw_complex* z, int incz)
{
  for(int i=0; i<n; i++) {
	 (*z)[0] += alpha * ( (*x)[0] * (*y)[0] - (*x)[1] * (*y)[1]);
	 (*z)[1] += alpha * ( (*x)[0] * (*y)[1] + (*x)[1] * (*y)[0]);
	 x = x + incx;
	 y = y + incy;
	 z = z + incz;
  }  
  return 0;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
vector<MatMgnt3d_MPI> MatMgnt3d_MPI::_mmvec;
/* For a given Kernel3d_MPI, knl, and level of prcision, np
 * see if we have already created that MatMgnt3d_MPI and return it.
 * If not, create a new one and store it in the _mmvec vector
 * for possible future use.  This is helpful when multiple calls
 * are made for the creation of a MatMgnt3d_MPI since if it
 * already exists, we can save time and storage.
 */
MatMgnt3d_MPI* MatMgnt3d_MPI::getmmptr(Kernel3d_MPI knl, int np)
{
  /* See if it already exists */
  for(size_t i=0; i<_mmvec.size(); i++) {
	 if(_mmvec[i].knl()==knl && _mmvec[i].np()==np) {
		return &(_mmvec[i]);
	 }
  }

  /* If it doesn't exist, create it, store it and return a pointer to it */
  _mmvec.push_back( MatMgnt3d_MPI() );
  int last = _mmvec.size()-1;
  MatMgnt3d_MPI* tmp = &(_mmvec[last]); //get the last one
  tmp->knl() = knl;
  tmp->np() = np;
  tmp->setup();
  return tmp;
}
