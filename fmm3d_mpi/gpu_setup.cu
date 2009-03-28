#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cutil.h>

#include "../p3d/point3d.h"
#include "gpu_setup.h"

#define PI_4I 0.079577471F
//#define PI_4I 1.0F

#define CERR

#if 0
#if defined (__cplusplus)
extern "C" int MPI_Comm_rank (int, int*);
#define MPI_COMM_WORLD 0x44000000
#endif
#endif

#if defined (CERR)
static
void
CE (FILE* fp)
{
  cudaError_t C_E = cudaGetLastError ();
  if (C_E) {
    int rank;
    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int procnamelen;
    memset (procname, 0, sizeof (procname));
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name (procname, &procnamelen);
    fprintf ((fp), "*** [%s:%lu::p%d(%s)] CUDA ERROR: %s ***\n", __FILE__, __LINE__, rank, procname, cudaGetErrorString (C_E));
    fflush (fp);
  }
}
#else
# define CE(fp)
#endif

size_t
gpu_count (void)
{
 int dev_count;
 CUDA_SAFE_CALL (cudaGetDeviceCount (&dev_count)); CE(stdout);
  if (dev_count > 0) {
    fprintf (stderr, "==> Found %d GPU device%s.\n",
   dev_count,
   dev_count == 1 ? "" : "s");
    return (size_t)dev_count;
  }
  return 0; /* no devices found */
}

void
gpu_dumpinfo (size_t dev_id)
{
  cudaDeviceProp p;
  assert (dev_id < gpu_count ());
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&p, (int)dev_id)); CE(stdout);
  fprintf (stderr, "==> Device %lu: \"%s\"\n", (unsigned long)dev_id, p.name);
  fprintf (stderr, " Major revision number: %d\n", p.major);
  fprintf (stderr, " Minor revision number: %d\n", p.minor);
  fprintf (stderr, " Total amount of global memory: %u MB\n", p.totalGlobalMem >> 20);
#if CUDART_VERSION >= 2000
  fprintf (stderr, " Number of multiprocessors: %d\n", p.multiProcessorCount);
  fprintf (stderr, " Number of cores: %d\n", 8 * p.multiProcessorCount);
#endif
  fprintf (stderr, " Total amount of constant memory: %u MB\n", p.totalConstMem >> 20);
  fprintf (stderr, " Total amount of shared memory per block: %u KB\n", p.sharedMemPerBlock >> 10);
  fprintf (stderr, " Total number of registers available per block: %d\n", p.regsPerBlock);
  fprintf (stderr, " Warp size: %d\n", p.warpSize);
  fprintf (stderr, " Maximum number of threads per block: %d\n", p.maxThreadsPerBlock);
  fprintf (stderr, " Maximum sizes of each dimension of a block: %d x %d x %d\n",
   p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
  fprintf (stderr, " Maximum sizes of each dimension of a grid: %d x %d x %d\n",
   p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
  fprintf (stderr, " Maximum memory pitch: %u bytes\n", p.memPitch);
  fprintf (stderr, " Texture alignment: %u bytes\n", p.textureAlignment);
  fprintf (stderr, " Clock rate: %.2f GHz\n", p.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
  fprintf (stderr, " Concurrent copy and execution: %s\n", p.deviceOverlap ? "Yes" : "No");
#endif
}

void
gpu_select (size_t dev_id)
{
  fprintf (stderr, "==> Selecting GPU device: %lu\n", (unsigned long)dev_id);
  CUDA_SAFE_CALL (cudaSetDevice ((int)dev_id)); CE(stdout);
  gpu_dumpinfo (dev_id);
}

//extern "C"
//void dense_inter_gpu(point3d_t*);
/*
*  3.tbdsf has the foll in order (dev mem):
*
*    start of cs and cp index
*
*    cl1 tot src boxes
*    cl2  tot num src
*  4.tbdsr has the foll
*    start of tsdf index
*    trgBoxSize
*    tp
*/


#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 1
//#define GRID_WIDTH 1

using namespace std;
////////////////////////////////////////BEGIN KERNEL///////////////////////////////////////////////
#ifdef DS_ORG
__global__ void ulist_kernel(float *t_dp,float *trgVal_dp,
          float *s_dp,
          int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
          int numAugTrg) {
#else
__global__ void ulist_kernel(float *tx_dp,float *ty_dp,float *tz_dp,float *trgVal_dp,
          float *sx_dp,float *sy_dp,float *sz_dp,float *srcDen_dp,
          int *tbdsr_dp,int *tbdsf_dp,int *cs_dp,int *cp_dp,
          int numAugTrg) {
#endif

#ifdef DS_ORG
  __shared__ float4 s_sh[BLOCK_HEIGHT];
  float3 t_reg;
#else
  __shared__ float sx_sh[BLOCK_HEIGHT];
  __shared__ float sy_sh[BLOCK_HEIGHT];
  __shared__ float sz_sh[BLOCK_HEIGHT];
  __shared__ float sd_sh[BLOCK_HEIGHT];
  float tx_reg,ty_reg,tz_reg;
#endif


//  __shared__ int cs_sh[];  //TODO: dynamic alloc
//  __shared__ int cp_sh[];


  int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
  if(uniqueBlockId<numAugTrg) {


    float tv_reg=0.0F;

    int boxId=tbdsr_dp[uniqueBlockId*3]*2;

    int trgLimit=tbdsr_dp[uniqueBlockId*3+1];
    int trgIdx=tbdsr_dp[uniqueBlockId*3+2]+threadIdx.x;  //can simplify by adding boxid to tbds base to make new pointer

  #ifdef DS_ORG
      t_reg=((float3*)t_dp)[trgIdx];
  #else
      tx_reg=tx_dp[trgIdx];
      ty_reg=ty_dp[trgIdx];
      tz_reg=tz_dp[trgIdx];
  #endif

  //  }

      float dX_reg;
      float dY_reg;
      float dZ_reg;



    int offset_reg=tbdsf_dp[boxId];
    int numSrc_reg=tbdsf_dp[boxId+1];
    int cs_idx_reg=0;

    int *cp_sh=cp_dp+offset_reg;    //TODO: fix this
    int *cs_sh=cs_dp+offset_reg;
    int loc_reg=cp_sh[0]+threadIdx.x;
    int num_thread_reg=threadIdx.x;
    int lastsum=cs_sh[0];

    //fetching cs and cp into shared mem
  //    for(int i=0;i<ceilf((float)numSrcBox_reg/(float)BLOCK_HEIGHT);i++)
  //      if(threadIdx.x<numSrcBox_reg-i*BLOCK_HEIGHT) {
  //        cs_sh[i*BLOCK_HEIGHT+threadIdx.x]=cs_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //        cp_sh[i*BLOCK_HEIGHT+threadIdx.x]=cp_dp[offset_reg+i*BLOCK_HEIGHT+threadIdx.x];
  //      }


    int num_chunk_loop=numSrc_reg/BLOCK_HEIGHT;

    for(int chunk=0;chunk<num_chunk_loop;chunk++) {


      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
        lastsum=cs_sh[cs_idx_reg];
      }

      __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((float4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

      loc_reg+=BLOCK_HEIGHT;
      num_thread_reg+=BLOCK_HEIGHT;

      __syncthreads();
#pragma unroll 32
      for(int src=0;src<BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
        dX_reg=s_sh[src].x-t_reg.x;
        dY_reg=s_sh[src].y-t_reg.y;
        dZ_reg=s_sh[src].z-t_reg.z;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*s_sh[src].w;
  #else
        dX_reg=sx_sh[src]-tx_reg;
        dY_reg=sy_sh[src]-ty_reg;
        dZ_reg=sz_sh[src]-tz_reg;

        dX_reg*=dX_reg;
        dY_reg*=dY_reg;
        dZ_reg*=dZ_reg;

        dX_reg += dY_reg+dZ_reg;

        dX_reg = rsqrtf(dX_reg);

        dX_reg = dX_reg + (dX_reg-dX_reg);
        dX_reg = fmaxf(dX_reg,0.0F);

        tv_reg+=dX_reg*sd_sh[src] ;
  #endif

        }
    }
    if(num_thread_reg<numSrc_reg) {
      if(num_thread_reg>=lastsum) {
        while(num_thread_reg>=cs_sh[cs_idx_reg]) cs_idx_reg++;
        loc_reg=cp_sh[cs_idx_reg]+(num_thread_reg-cs_sh[cs_idx_reg-1]);
  //      lastsum=cs_sh[cs_idx_reg];
      }
    }
    __syncthreads();
  #ifdef DS_ORG
      s_sh[threadIdx.x]=((float4*)s_dp)[loc_reg];
  #else
      sx_sh[threadIdx.x]=sx_dp[loc_reg];
      sy_sh[threadIdx.x]=sy_dp[loc_reg];
      sz_sh[threadIdx.x]=sz_dp[loc_reg];
      sd_sh[threadIdx.x]=srcDen_dp[loc_reg];
  #endif

    __syncthreads();

    for(int src=0;src<numSrc_reg%BLOCK_HEIGHT;src++) {
  #ifdef DS_ORG
      dX_reg=s_sh[src].x-t_reg.x;
      dY_reg=s_sh[src].y-t_reg.y;
      dZ_reg=s_sh[src].z-t_reg.z;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);
      tv_reg+=dX_reg*s_sh[src].w;
  #else
      dX_reg=sx_sh[src]-tx_reg;
      dY_reg=sy_sh[src]-ty_reg;
      dZ_reg=sz_sh[src]-tz_reg;

      dX_reg*=dX_reg;
      dY_reg*=dY_reg;
      dZ_reg*=dZ_reg;

      dX_reg += dY_reg+dZ_reg;

      dX_reg = rsqrtf(dX_reg);

      dX_reg = dX_reg + (dX_reg-dX_reg);
      dX_reg = fmaxf(dX_reg,0.0F);

      tv_reg+=dX_reg*sd_sh[src] ;
  #endif

    }


    if(threadIdx.x<trgLimit) {
      trgVal_dp[trgIdx]=tv_reg*PI_4I;    //div by pi here not inside loop
    }

  }    //extra invalid padding block


}















void make_ds(int **tbdsf, int **tbdsr, int **cs, int **cp, point3d_t* P,int *numAugTrg,int *numSrcBoxTot) {
  for(int i=0;i<P->numTrgBox;i++) {
    *numAugTrg+=(P->trgBoxSize[i]/BLOCK_HEIGHT+((P->trgBoxSize[i]%BLOCK_HEIGHT)?1:0));
    *numSrcBoxTot+=P->uListLen[i];

  }
  int srcidx[P->numSrcBox];
  int srcsum=0;
  for(int i=0;i<P->numSrcBox;i++) {
    srcidx[i]=srcsum;
    srcsum+=P->srcBoxSize[i];
  }
//  cout<<"Split "<<P->numTrgBox<<" targets boxes into "<<*numAugTrg<<endl;
//  cout<<"Total source boxes: "<<*numSrcBoxTot<<endl;

  *tbdsf=(int*)malloc(sizeof(int)*2*P->numTrgBox);
  *tbdsr=(int*)malloc(sizeof(int)*3**numAugTrg);
  *cs=(int*)malloc(sizeof(int)**numSrcBoxTot);
  *cp=(int*)malloc(sizeof(int)**numSrcBoxTot);

  int cc=0;
  int tt=0;
  int tbi=0;

  for(int i=0;i<P->numTrgBox;i++) {
    (*tbdsf)[i*2]=cc;
    int cumulSum=0;
    for(int k=0;k<P->uListLen[i];k++) {
      int srcbox=P->uList[i][k];
      cumulSum+=P->srcBoxSize[srcbox];
      (*cs)[cc]=cumulSum;
      (*cp)[cc]=srcidx[srcbox];
      cc++;
    }
    (*tbdsf)[i*2+1]=cumulSum;
    int remtrg=P->trgBoxSize[i];
    while(remtrg>0) {
      (*tbdsr)[3*tbi]=i;
      (*tbdsr)[3*tbi+1]=(remtrg<BLOCK_HEIGHT)?remtrg:BLOCK_HEIGHT;
      (*tbdsr)[3*tbi+2]=tt;
      tt+=(*tbdsr)[3*tbi+1];
      tbi++;    //tbi corresponds to gpu block id
      remtrg-=BLOCK_HEIGHT;
    }
  }
}

//extern "C"
//{
void dense_inter_gpu(point3d_t *P) {
  //Initialize device
//  int devID;
//  devID = cutGetMaxGflopsDeviceId();
//  cudaSetDevice( /*devID*/ 0 ); CE(stdout)  //done: Fix this for multiple devices.. maxgflops
//  unsigned int timer;
//  float ms;
//  cutCreateTimer(&timer);
#ifdef DS_ORG
  float *s_dp,*t_dp;
#else
  float *sx_dp,*sy_dp,*sz_dp,*tx_dp,*ty_dp,*tz_dp,*srcDen_dp;
#endif
  float *trgVal_dp;
  int *tbdsf_dp, *tbdsr_dp;
  int *tbdsf,*tbdsr,*cs,*cp,numAugTrg=0,numSrcBoxTot=0;
  int *cs_dp,*cp_dp;
//  cutResetTimer(timer);
//  CUT_SAFE_CALL(cutStartTimer(timer));
  make_ds(&tbdsf,&tbdsr,&cs,&cp,P,&numAugTrg,&numSrcBoxTot);        //done: does not exist yet
//  CUT_SAFE_CALL(cutStopTimer(timer));
//   ms = cutGetTimerValue(timer);
//   cout<<"Preparing data structures: "<<ms<<"ms"<<endl;

//  for(int i=0;i<numSrcBoxTot;i++) {
//    cout<<cs[i]<<" ";
//  }
//   cutResetTimer(timer);
//  CUT_SAFE_CALL(cutStartTimer(timer));
#ifdef DS_ORG
  cudaMalloc((void**)&s_dp,P->numSrc*sizeof(float)*4); CE(stdout);  //float4

  cudaMalloc((void**)&t_dp,P->numTrg*sizeof(float)*3); CE(stdout);  //float3
#else
  cudaMalloc((void**)&sx_dp,P->numSrc*sizeof(float));
  cudaMalloc((void**)&sy_dp,P->numSrc*sizeof(float));
  cudaMalloc((void**)&sz_dp,P->numSrc*sizeof(float));

  cudaMalloc((void**)&tx_dp,P->numTrg*sizeof(float));
  cudaMalloc((void**)&ty_dp,P->numTrg*sizeof(float));
  cudaMalloc((void**)&tz_dp,P->numTrg*sizeof(float));

  cudaMalloc((void**)&srcDen_dp,P->numSrc*sizeof(float));
#endif



  cudaMalloc((void**)&trgVal_dp,P->numTrg*sizeof(float)); CE(stdout);

  cudaMalloc((void**)&tbdsf_dp,P->numTrgBox*2*sizeof(int)); CE(stdout);

  cudaMalloc((void**)&tbdsr_dp,3*numAugTrg*sizeof(int)); CE(stdout);

  cudaMalloc((void**)&cs_dp,numSrcBoxTot*sizeof(int)); CE(stdout);

  cudaMalloc((void**)&cp_dp,numSrcBoxTot*sizeof(int)); CE(stdout);


  //Put data into the device
#ifdef DS_ORG
  cudaMemcpy(s_dp,P->src_,P->numSrc*sizeof(float)*4,cudaMemcpyHostToDevice); CE(stdout);

  cudaMemcpy(t_dp,P->trg_,P->numTrg*sizeof(float)*3,cudaMemcpyHostToDevice); CE(stdout);
#else
  cudaMemcpy(sx_dp,P->sx_,P->numSrc*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);
  cudaMemcpy(sy_dp,P->sy_,P->numSrc*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);
  cudaMemcpy(sz_dp,P->sz_,P->numSrc*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);

  cudaMemcpy(tx_dp,P->tx_,P->numTrg*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);
  cudaMemcpy(ty_dp,P->ty_,P->numTrg*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);
  cudaMemcpy(tz_dp,P->tz_,P->numTrg*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);

  cudaMemcpy(srcDen_dp,P->srcDen,P->numSrc*sizeof(float),cudaMemcpyHostToDevice); CE(stdout);
#endif

  cudaMemcpy(tbdsf_dp,tbdsf,2*sizeof(int)*P->numTrgBox,cudaMemcpyHostToDevice); CE(stdout);

  cudaMemcpy(tbdsr_dp,tbdsr,3*sizeof(int)*numAugTrg,cudaMemcpyHostToDevice); CE(stdout);


  cudaMemcpy(cs_dp,cs,(numSrcBoxTot+1)*sizeof(int),cudaMemcpyHostToDevice); CE(stdout);

  cudaMemcpy(cp_dp,cp,(numSrcBoxTot+1)*sizeof(int),cudaMemcpyHostToDevice); CE(stdout);


  //kernel call
  int GRID_WIDTH=(int)ceil((float)numAugTrg/65535.0F);
  int GRID_HEIGHT=(int)ceil((float)numAugTrg/(float)GRID_WIDTH);
//  cout<<"Width: "<<GRID_WIDTH<<" HEIGHT: "<<GRID_HEIGHT<<endl;
//  cout<<"Number of gpu blocks: "<<numAugTrg<<endl;
  dim3 BlockDim(BLOCK_HEIGHT,BLOCK_WIDTH);  //Block width will be 1
  dim3 GridDim(GRID_HEIGHT, GRID_WIDTH);    //Grid width should be 1

//  for(int i=0;i<P->numTrg;i++) {
//    P->trgVal[i]=0.0F;
//  }
//  cout<<"Kernel call: ";
#ifdef DS_ORG
  ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg); CE(stdout);
#else
  ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(tx_dp,ty_dp,tz_dp,trgVal_dp,sx_dp,sy_dp,sz_dp,srcDen_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg); CE(stdout);
#endif

        cudaMemcpy(P->trgVal,trgVal_dp,sizeof(float)*P->numTrg,cudaMemcpyDeviceToHost); CE(stdout);
//  CUT_SAFE_CALL(cutStopTimer(timer));
//   ms = cutGetTimerValue(timer);
//   cout<<ms<<"ms"<<endl;

//  for(int i=0;i<100;i++) {
//    cout<<tbdsf[i*3+2]<<" ";
//  }
#ifdef DS_ORG
  cudaFree(s_dp);
  cudaFree(t_dp);
#else
  cudaFree(sx_dp);
  cudaFree(sy_dp);
  cudaFree(sz_dp);

  cudaFree(tx_dp);
  cudaFree(ty_dp);
  cudaFree(tz_dp);

  cudaFree(srcDen_dp);
#endif

  cudaFree(trgVal_dp);
  cudaFree(tbdsf_dp);
  cudaFree(tbdsr_dp);
  cudaFree(cs_dp);
  cudaFree(cp_dp);

  free(cs);
  free(cp);
  free(tbdsf);
  free(tbdsr);
}
//}//end extern
