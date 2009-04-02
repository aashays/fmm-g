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
#include <cassert>
#include <cstring>
#include "fmm3d_mpi.hpp"
#include "manage_petsc_events.hpp"
#include "sys/sys.h"
#include "parUtils.h"
#ifdef HAVE_TAU
#include <Profile/Profiler.h>
#endif
#include <cstring>

#include "gpu_setup.h"

using namespace std;

/*! For a certain processor, return local number of positions from Vec pos 
 * See http://www-unix.mcs.anl.gov/petsc/petsc-as/snapshots/petsc-current/docs/manualpages/Vec/VecGetLocalSize.html for more info */
PetscInt  procLclNum(Vec pos) { PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
/*! For a certain processor, return global number of positions from Vec pos */
PetscInt  procGlbNum(Vec pos) { PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }

inline double gaussian(double mean, double std_deviation) {
  static double t1 = 0, t2=0;
  double x1, x2, x3, r;

  using namespace std;

  // reuse previous calculations
  if(t1) {
    const double tmp = t1;
    t1 = 0;
    return mean + std_deviation * tmp;
  }
  if(t2) {
    const double tmp = t2;
    t2 = 0;
    return mean + std_deviation * tmp;
  }

  // pick randomly a point inside the unit disk
  do {
    x1 = 2 * drand48() - 1;
    x2 = 2 * drand48() - 1;
    x3 = 2 * drand48() - 1;
    r = x1 * x1 + x2 * x2 + x3*x3;
  } while(r >= 1);

  // Box-Muller transform
  r = sqrt(-2.0 * log(r) / r);

  // save for next call
  t1 = (r * x2);
  t2 = (r * x3);

  return mean + (std_deviation * r * x1);
}


int main(int argc, char** argv)
{
#ifdef HAVE_TAU
  TAU_PROFILE_TIMER(tau_eval_timer,"fmm eval", "void (void)", TAU_USER);
  // TAU_PROFILE("int main(int, char **)", " ", TAU_DEFAULT);
  TAU_INIT(&argc, &argv); 
#ifndef TAU_MPI
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
#endif
  PetscInitialize(&argc,&argv,NULL,NULL); 
  ot::RegisterEvents();
  registerPetscEvents();

  PetscTruth per_core_summary;
  char per_core_summ_file_suffix[1024];
  PetscOptionsGetString(0,"-per_core_summary",per_core_summ_file_suffix,1024/*length*/,&per_core_summary);

  // Initialize GPU device
  PetscTruth gpu_ulist;
  PetscOptionsHasName(0,"-gpu_ulist",&gpu_ulist);
  if (gpu_ulist) {
#ifdef COMPILE_GPU
    size_t num_gpus = gpu_count ();
    if (!num_gpus) {
      cerr << "*** ERROR: No GPU devices available! ***" << endl;
      return -1;
    }
    for (size_t dev_id = 0; dev_id < num_gpus; ++dev_id)
      gpu_dumpinfo (dev_id);

    int mpirank;
    MPI_Comm_rank (PETSC_COMM_WORLD, &mpirank);

    assert (num_gpus);
    int gpu_id = mpirank % num_gpus;

    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int len;
    memset (procname, 0, sizeof (procname));
    MPI_Get_processor_name (procname, &len);

    gpu_select (gpu_id);
    cout << "==> p" << mpirank << "(" << procname << ") --> GPU #" << gpu_id << endl;
#else
    SETERRQ(1,"GPU code not compiled");
#endif
  }

  // make "Main Stage" invisible
  StageLog CurrentStageLog;
  PetscLogGetStageLog(&CurrentStageLog);
  int CurrentStage;
  StageLogGetCurrent(CurrentStageLog, &CurrentStage);
  pA(CurrentStage!=-1);
  StageLogSetVisible(CurrentStageLog, CurrentStage, PETSC_FALSE);
  
  // previously per-core summary did not seem to work, if we did not execute here "PetscLogBegin" (something went wrong in DumpPerCoreSummary() )
  // now it seems to work fine without the call below 
  // if (per_core_summary)
  //   PetscLogBegin();
  for (int preloading=1; preloading>=0; preloading--)
  {
    if (!preloading)
      PetscLogStagePush(stages[0]);

    MPI_Comm comm; comm = PETSC_COMM_WORLD;
    int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if (!mpirank) {
      if (preloading)
	cout<<endl<<"*** Dry run (preloading) ***"<<endl;
      else
	cout<<endl<<"*** Final run ***"<<endl;
    }

    int dim = 3;
    srand48( mpirank );

    PetscTruth flg = PETSC_FALSE;

    //1. allocate random data
    PetscInt numsrc;  
    pC( PetscOptionsGetInt(preloading? "preload_" : 0, "-numsrc", &numsrc, &flg) );
    pA(flg==PETSC_TRUE);

    if (!mpirank)
      cout<<"Using numsrc=" << numsrc << endl;

    PetscInt kt;
    pC( PetscOptionsGetInt(0, "-kt", &kt, &flg) );
    pA(flg==PETSC_TRUE);

    PetscTruth doing_isogranular;
    PetscOptionsHasName(0,"-isogranular",&doing_isogranular);

    const int distLen=50;
    char distribution[distLen];
    {
      PetscTruth distribution_flg;
      PetscOptionsGetString(0,"-distribution",distribution,distLen,&distribution_flg);
      assert(distribution_flg);
    }

    vector<double> tmp(2); 
    tmp[0] = 1;
    tmp[1] = 0.25; //coefs in the kernel, work for all examples

    Kernel3d_MPI knl(kt, tmp);

    PetscInt lclnumsrc;

    if (preloading || doing_isogranular)
      // interpret "-numsrc" as per-processor number of sources
    {
      if (!mpirank)
	cout<<"Interpreting -numsrc as per-processor"<<endl;
      lclnumsrc = numsrc;
    }
    else
      // otherwise interpret "-numsrc" as global number of sources
    {
      if (!mpirank)
	cout<<"Interpreting -numsrc as global (NOT per-processor)"<<endl;
      lclnumsrc = (numsrc+mpirank)/mpisize;
    }

    // R,r, lclctr are used when non-uniform distribution from former "tt1.cpp" is used (boolean flag "use_tt1_distr"). Otherwise R,r,lclctr are ignored.
    //pick center
    double R = 0.25;
    double r = 0.499-R;
    double lclctr[3];
    for(int d=0; d<dim; d++)	 lclctr[d] = 0.5 + R * (2.0*drand48()-1.0);

    // generate source positions
    vector<double> srcPosarr(lclnumsrc*dim);
    if (0==strcmp(distribution,"tt1"))
    {
      if (!mpirank)
	cout<<"Using distribution from tt1.cpp for sources"<<endl;

      for(PetscInt k=0; k<lclnumsrc; k++)
	for(int d=0; d<dim; d++)
	  srcPosarr[d+dim*k] = lclctr[d] + r*(2.0*drand48()-1.0);
    }
    else if (0==strcmp(distribution,"uniform"))
    {
      if (!mpirank)
	cout<<"Using uniform distribution for sources"<<endl;

      for(PetscInt k=0; k<lclnumsrc*dim; k++)
	srcPosarr[k] = drand48();
    }
    else if (0==strcmp(distribution,"normal"))
    {
      if (!mpirank)
	cout<<"Using normal distribution for sources"<<endl;

      for(PetscInt k=0; k<lclnumsrc; k++  )
      {
normal_skip_point:
	double tmp[3];
	for(int d=0; d<3; d++)
	  tmp[d]= gaussian(0.5, 0.16);
	for(int d=0; d<3; d++)
	  if (tmp[d]>=1.0 || tmp[d]<0 )
	    goto normal_skip_point;

	for(int d=0; d<3; d++)
	  srcPosarr[d+3*k] = tmp[d];
      }
    }
    else if (0==strcmp(distribution,"logNormal"))
    {
      if (!mpirank)
	cout<<"Using log-normal distribution for sources"<<endl;

      for(PetscInt k=0; k<lclnumsrc; k++  )
      {
lognormal_skip_point:
	double tmp[3];
	for(int d=0; d<3; d++)
	  tmp[d]= exp(gaussian(-1.75, 1.0));
	for(int d=0; d<3; d++)
	  if (tmp[d]>=1.0)
	    goto lognormal_skip_point;

	for(int d=0; d<3; d++)
	  srcPosarr[d+3*k] = tmp[d];
      }
    }
    else if (0==strcmp(distribution,"sphereUniformAngles"))
    {
      if (!mpirank)
	cout<<"Using distribution on sphere, uniform in angles"<<endl;
      for(PetscInt k=0; k<lclnumsrc; k++)
      {
	const double r=0.49;
	const double center [3] = { 0.5, 0.5, 0.5};
	double phi=2*M_PI*drand48();
	double theta=M_PI*drand48();
	srcPosarr[0+3*k]=center[0]+r*sin(theta)*cos(phi);
	srcPosarr[1+3*k]=center[1]+r*sin(theta)*sin(phi);
	srcPosarr[2+3*k]=center[2]+r*cos(theta);
      }
    }
    else if (0==strcmp(distribution,"ellipseUniformAngles"))
      // x:y:z will be 1:1:4; that is, points will cluster at "poles" (x=0,y=0,z=+-1), where the curvature is max.
    {
      if (!mpirank)
	cout<<"Using distribution on 1:1:4 ellipse, uniform in angles"<<endl;
      for(PetscInt k=0; k<lclnumsrc; k++)
      {
	const double r=0.49;
	const double center [3] = { 0.5, 0.5, 0.5};
	double phi=2*M_PI*drand48();
	double theta=M_PI*drand48();
	srcPosarr[0+3*k]=center[0]+0.25*r*sin(theta)*cos(phi);
	srcPosarr[1+3*k]=center[1]+0.25*r*sin(theta)*sin(phi);
	srcPosarr[2+3*k]=center[2]+r*cos(theta);
      }
    }
    else
      SETERRQ(1,"Invalid distribution of sources selected");

    Vec srcPos;
    // now sort the sources and remove duplicates unless instructed not to
    PetscTruth skip_initial_sort;
    PetscOptionsHasName(0,"-skip_initial_sort",&skip_initial_sort);
    if (!skip_initial_sort)
    {
      if(!mpirank)
	cout<<"Sorting the sources in morton order and removing duplicates"<<endl;
      const unsigned maxDepth = 30;
      std::vector<ot::TreeNode> tmpNodes(lclnumsrc);
      for (int i=0; i<lclnumsrc; i++)
      {
	unsigned X = unsigned( ldexp(srcPosarr[3*i],maxDepth) );
	unsigned Y = unsigned( ldexp(srcPosarr[3*i+1],maxDepth) );
	unsigned Z = unsigned( ldexp(srcPosarr[3*i+2],maxDepth) );
	tmpNodes[i]=ot::TreeNode(X,Y,Z,maxDepth,3,maxDepth);
      }
      par::removeDuplicates<ot::TreeNode>(tmpNodes,false,comm);	
      lclnumsrc = tmpNodes.size();
      srcPosarr.resize(lclnumsrc*dim);
      for (int i=0; i<lclnumsrc; i++)
      {
	srcPosarr[0+3*i]=ldexp(tmpNodes[i].getX()+0.5,-maxDepth);
	srcPosarr[1+3*i]=ldexp(tmpNodes[i].getY()+0.5,-maxDepth);
	srcPosarr[2+3*i]=ldexp(tmpNodes[i].getZ()+0.5,-maxDepth);
      }
    }

    // when this vector is destroyed, srcPosarr won't be freed; srcPosarr will be freed by it's destructor in the end of its scope
    VecCreateMPIWithArray(comm, lclnumsrc*dim, PETSC_DETERMINE, srcPosarr.size()? &srcPosarr[0]:0, &srcPos);

    if(!mpirank)
      cout<<"Total number of sources on all processors: "<<procGlbNum(srcPos) << endl;

    {
      // if instructed, dump vector of source coordinates to a file
      PetscTruth dump_sources;
      const int fNameLen=1000;
      char fName[fNameLen];
      PetscOptionsGetString(preloading? "preload_": 0,"-dump_src_coord",fName,fNameLen,&dump_sources);
      if (dump_sources)
      {
	if (!mpirank)
	  cout<<"Dumping source coordinates to file"<<endl;
	PetscViewer viewer;
	PetscViewerASCIIOpen (comm,fName,&viewer);
	PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
	VecView(srcPos,viewer);
	PetscViewerDestroy(viewer);
      }
    }

    // create vector of normals; looks like it is just ignored for single-layer kernels 
    Vec srcNor;  pC( VecCreateMPI(comm, lclnumsrc*dim, PETSC_DETERMINE, &srcNor) );
    double* srcNorarr; pC( VecGetArray(srcNor, &srcNorarr) );
    for(PetscInt k=0; k<lclnumsrc; k++){
      srcNorarr[3*k] = drand48();
      srcNorarr[3*k+1] = drand48();
      srcNorarr[3*k+2] = drand48();
      double norm=sqrt(srcNorarr[3*k]*srcNorarr[3*k] + srcNorarr[3*k+1]*srcNorarr[3*k+1] + srcNorarr[3*k+2]*srcNorarr[3*k+2]);
      srcNorarr[3*k] /= norm;
      srcNorarr[3*k+1] /= norm;
      srcNorarr[3*k+2] /= norm;
    }
    pC( VecRestoreArray(srcNor, &srcNorarr) );


    Vec trgPos;  
    PetscTruth trgs_and_srcs_coinc;
    PetscOptionsHasName(0,"-trgs_and_srcs_coinc",&trgs_and_srcs_coinc);
    if(trgs_and_srcs_coinc)
    {
      if (!mpirank)
	cout<<"Targets and sources coincide"<<endl;
      trgPos=srcPos; // we do assume petsc type "Vec" is actually a pointer 
    }
    else
    {
      PetscInt numtrg;
      pC( PetscOptionsGetInt(preloading? "preload_": 0, "-numtrg", &numtrg, &flg) );
      pA(flg==PETSC_TRUE);

      PetscInt lclnumtrg;
      if (preloading || doing_isogranular)
	// interpret "-numtrg" as per-processor number of targets
      {
	if (!mpirank)
	  cout<<"Interpreting -numtrg as per-processor"<<endl;
	lclnumtrg = numtrg;
      }
      else
	// otherwise interpret "-numtrg" as global number of targets
      {
	if (!mpirank)
	  cout<<"Interpreting -numtrg as global (NOT per-processor)"<<endl;
	lclnumtrg = (numtrg+mpirank)/mpisize;
      }

      if (!mpirank)
	cout<<"Using numtrg=" << numtrg << endl;

      pC( VecCreateMPI(comm, lclnumtrg*dim, PETSC_DETERMINE, &trgPos) );
      double* trgPosarr; pC( VecGetArray(trgPos, &trgPosarr) );

      if (0==strcmp(distribution,"tt1"))
      {
	if (!mpirank)
	  cout<<"Using distribution from tt1.cpp for targets"<<endl;
	for(PetscInt k=0; k<lclnumtrg; k++)
	  for(int d=0; d<dim; d++)
	    trgPosarr[d+dim*k] = lclctr[d] + r*(2.0*drand48()-1.0);
      }
      else if (0==strcmp(distribution,"uniform"))
      {
	if (!mpirank)
	  cout<<"Using uniform distribution for targets"<<endl;

	for(PetscInt k=0; k<lclnumtrg*dim; k++)
	  trgPosarr[k] = drand48();
      }
      else
	SETERRQ(1,"Invalid distribution of sources/targets selected");
      pC( VecRestoreArray(trgPos, &trgPosarr) );
    }

    if (!mpirank)
      cout<<"Total number of targets on all processors: "<<procGlbNum(trgPos) << endl;

    //2. allocate fmm 
    FMM3d_MPI* fmm = new FMM3d_MPI("fmm3d_");
    fmm->srcPos()=srcPos;
    fmm->srcNor()=srcNor;
    fmm->trgPos()=trgPos;

    // we shall use the box [0,1]^3
    fmm->ctr() = Point3(0.5,0.5,0.5); // CENTER OF THE TOPLEVEL BOX
    fmm->rootLevel() = 1;         // 2^(-rootlvl) is the RADIUS OF THE TOPLEVEL BOX
    fmm->knl() = knl;

    if (!preloading)
    {
      PetscLogStagePop();
      PetscLogStagePush(stages[1]);
    }

    // PreLoadStage("FMM_Setup");

    // setup destroys srcPos, srcNor, trgPos (after creating new, redistributed ones) 
    MPI_Barrier(comm);
    PetscLogEventBegin(fmm_setup_event,0,0,0,0);
    pC( fmm->setup() );
    PetscLogEventEnd(fmm_setup_event,0,0,0,0);

    if (!preloading)
    {
      PetscLogStagePop();
      PetscLogStagePush(stages[2]);
    }
    // PreLoadStage("GenSrcDen");
    // now sources and targets are re-distributed, pointers srcPos, trgPos, srcNor are invalid
    srcPos=fmm->srcPos();
    trgPos=fmm->trgPos();
    srcNor=fmm->srcNor();

    lclnumsrc = procLclNum(srcPos);
    PetscInt lclnumtrg = procLclNum(trgPos);

    int srcDOF = knl.srcDOF();
    int trgDOF = knl.trgDOF();
    if (!mpirank)
      cout<<"srcDOF="<<srcDOF<<" trgDOF="<<trgDOF<<endl;

    Vec srcDen;  pC( VecCreateMPI(comm, lclnumsrc*srcDOF, PETSC_DETERMINE, &srcDen) );
    double* srcDenarr; pC( VecGetArray(srcDen, &srcDenarr) );
    for(PetscInt k=0; k<lclnumsrc*srcDOF; k++)
      srcDenarr[k] = drand48();
    pC( VecRestoreArray(srcDen, &srcDenarr) );

    Vec trgVal;  pC( VecCreateMPI(comm, lclnumtrg*trgDOF, PETSC_DETERMINE, &trgVal) );
    pC( VecSet(trgVal, 0.0) );

    //3. run fmm 
    if (!preloading)
    {
      PetscLogStagePop();
      PetscLogStagePush(stages[3]);
    }

    MPI_Barrier(comm);
    PetscLogEventBegin(fmm_eval_event,0,0,0,0);
#ifdef HAVE_TAU
    if (!preloading)
      TAU_PROFILE_START(tau_eval_timer);
#endif
    pC( fmm->evaluate(srcDen, trgVal) );
#ifdef HAVE_TAU
    if (!preloading)
      TAU_PROFILE_STOP(tau_eval_timer);
#endif
    PetscLogEventEnd(fmm_eval_event,0,0,0,0);

    // second evaluation (now some matrices are precomputed, so timing might be different)
    if (!preloading)
    {
      PetscLogStagePop();
      PetscLogStagePush(stages[4]);
    }

    PetscTruth do2ndEval;
    PetscOptionsHasName(0,"-do_2nd_eval",&do2ndEval);
    if (do2ndEval)
    {
      MPI_Barrier(comm);
      PetscLogEventBegin(fmm_eval_event,0,0,0,0);
#ifdef HAVE_TAU
      if (!preloading)
	TAU_PROFILE_START(tau_eval_timer);
#endif
      pC( fmm->evaluate(srcDen, trgVal) );
#ifdef HAVE_TAU
      if (!preloading)
	TAU_PROFILE_STOP(tau_eval_timer);
#endif
      PetscLogEventEnd(fmm_eval_event,0,0,0,0);
    }

    //4. check
    if (!preloading)
    {
      PetscLogStagePop();
      PetscLogStagePush(stages[5]);
    }
    // PreLoadStage("FMM_check");
    PetscInt lclNumChk;

    PetscTruth check_all;
    PetscOptionsHasName(preloading? "preload_": 0,"-check_all",&check_all);
    if (!check_all)
    {
      pC( PetscOptionsGetInt(preloading? "preload_": 0, "-numchk", &lclNumChk, &flg) );
      pA(flg==PETSC_TRUE);
    }
    else
      lclNumChk=lclnumtrg; // check all local targets

    PetscInt glbNumChk;
    MPI_Allreduce ( &lclNumChk, &glbNumChk, 1, MPIU_INT, MPI_SUM, comm);
    if (glbNumChk)
    {
      if(!mpirank)
	cout<<"Peeking globally "<<glbNumChk<<" target points (out of "<<procGlbNum(trgPos)<<" ) to check potential"<<endl;

      double rerr;
      MPI_Barrier(comm);
      PetscLogEventBegin(fmm_check_event,0,0,0,0);
      pC( fmm->check(srcDen, trgVal, lclNumChk, rerr) );
      PetscLogEventEnd(fmm_check_event,0,0,0,0);
      pC( PetscPrintf(MPI_COMM_WORLD, "Relative error is: %e\n", rerr) );
    }
    else
      if(!mpirank)
	cout<<"No points to check potential, thus skipping check."<<endl;

    delete fmm;

    pC( VecDestroy(srcPos) );
    pC( VecDestroy(srcNor) );
    if (!trgs_and_srcs_coinc)
      pC( VecDestroy(trgPos) );
    pC( VecDestroy(srcDen) );
    pC( VecDestroy(trgVal) );

    if (!preloading)
      PetscLogStagePop();
  }

  if (per_core_summary)
    DumpPerCoreSummary(MPI_COMM_WORLD, per_core_summ_file_suffix);

  PetscFinalize();
  return 0;
}
