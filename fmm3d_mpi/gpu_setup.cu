#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <cuda.h>
#include <cutil.h>

#include "../p3d/point3d.h"
#include "gpu_setup.h"

#define PI_4I 0.079577471F
//#define PI_4I 1.0F

#if defined (GPU_CERR)
void
gpu_checkerr__stdout (const char* filename, size_t line)
{
  FILE* fp = stdout;
  cudaError_t C_E = cudaGetLastError ();
  if (C_E) {
    int rank;
    char procname[MPI_MAX_PROCESSOR_NAME+1];
    int procnamelen;
    memset (procname, 0, sizeof (procname));
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name (procname, &procnamelen);
    fprintf ((fp), "*** [%s:%lu--p%d(%s)] CUDA ERROR: %s ***\n", filename, line, rank, procname, cudaGetErrorString (C_E));
    fflush (fp);
  }
}
#endif

void
gpu_msg__stdout (const char* msg, const char* filename, size_t lineno)
{
  FILE* fp = stdout;
  int rank;
  char procname[MPI_MAX_PROCESSOR_NAME+1];
  int procnamelen;
  memset (procname, 0, sizeof (procname));
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name (procname, &procnamelen);
  fprintf (fp, "===> [%s:%lu--p%d(%s)] %s\n", filename, lineno, rank, procname, msg);
}

void
gpu_check_pointer (const void* p, const char* fn, size_t l)
{
  if (!p) {
    gpu_msg__stdout ("NULL pointer", fn, l);
    MPI_Abort (MPI_COMM_WORLD, -1);
    assert (p);
  }
}

size_t
gpu_count (void)
{
 int dev_count;
 CUDA_SAFE_CALL (cudaGetDeviceCount (&dev_count)); GPU_CE;
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
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&p, (int)dev_id)); GPU_CE;
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
  CUDA_SAFE_CALL (cudaSetDevice ((int)dev_id)); GPU_CE;
  gpu_dumpinfo (dev_id);
}

/** Allocates 'n' bytes, initialized to zero */
static
void *
gpu_calloc_ (size_t n)
{
  void* p;
  cudaMalloc(&p, n); GPU_CE;
  if (!p) {
    int mpirank;
    MPI_Comm_rank (MPI_COMM_WORLD, &mpirank);
    fprintf (stderr, "[%s:%lu::p%d] Can't allocate %lu bytes!\n",
	     __FILE__, __LINE__, mpirank, (unsigned long)n);
  }
  assert (p);
  cudaMemset (p, 0, n); GPU_CE;
  return p;
}

float *
gpu_calloc_float (size_t n)
{
  return (float *)gpu_calloc_ (n * sizeof (float));
}

int *
gpu_calloc_int (size_t n)
{
  return (int *)gpu_calloc_ (n * sizeof (int));
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

void
gpu_copy_cpu2gpu_float (float* d, const float* s, size_t n)
{
  cudaMemcpy (d, s, n * sizeof (float), cudaMemcpyHostToDevice);
  GPU_CE;
}

void
gpu_copy_cpu2gpu_int (int* d, const int* s, size_t n)
{
  if (n) {
    cudaMemcpy (d, s, n * sizeof (int), cudaMemcpyHostToDevice);
    GPU_CE;
  }
}

void
gpu_copy_gpu2cpu_float (float* d, const float* s, size_t n)
{
  cudaMemcpy (d, s, n * sizeof (float), cudaMemcpyDeviceToHost);
  GPU_CE;
}

////////////////////////////////////////BEGIN KERNEL///////////////////////////////////////////////

#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 1
//#define GRID_WIDTH 1

using namespace std;

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
    } // chunk
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

  *tbdsf=(int*)malloc(sizeof(int)*2*P->numTrgBox); assert (*tbdsf || !P->numTrgBox);
  *tbdsr=(int*)malloc(sizeof(int)*3**numAugTrg); assert (*tbdsr || !numAugTrg);
  *cs=(int*)malloc(sizeof(int)**numSrcBoxTot); assert (*cs || !numSrcBoxTot);
  *cp=(int*)malloc(sizeof(int)**numSrcBoxTot); assert (*cp || !numSrcBoxTot);

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
#ifdef DS_ORG
  float *s_dp,*t_dp;
#else
  float *sx_dp,*sy_dp,*sz_dp,*tx_dp,*ty_dp,*tz_dp,*srcDen_dp;
#endif
  float *trgVal_dp;
  int *tbdsf_dp, *tbdsr_dp;
  int *tbdsf,*tbdsr,*cs,*cp,numAugTrg=0,numSrcBoxTot=0;
  int *cs_dp,*cp_dp;

  GPU_MSG ("GPU U-list");

  make_ds (&tbdsf, &tbdsr, &cs, &cp, P, &numAugTrg, &numSrcBoxTot);

#ifdef DS_ORG
  s_dp = gpu_calloc_float ((P->numSrc + BLOCK_HEIGHT) * 4); /* Padded by BLOCK_HEIGHT */
  t_dp = gpu_calloc_float ((P->numTrg + BLOCK_HEIGHT) * 3);
#else
  sx_dp = gpu_calloc_float (P->numSrc);
  sy_dp = gpu_calloc_float (P->numSrc);
  sz_dp = gpu_calloc_float (P->numSrc);
  tx_dp = gpu_calloc_float (P->numTrg);
  ty_dp = gpu_calloc_float (P->numTrg);
  tz_dp = gpu_calloc_float (P->numTrg);
  srcDen_dp = gpu_calloc_float (P->numSrc);
#endif

  trgVal_dp = gpu_calloc_float (P->numTrg);
  tbdsf_dp = gpu_calloc_int (P->numTrgBox * 2);
  tbdsr_dp = gpu_calloc_int (numAugTrg * 3);
  cs_dp = gpu_calloc_int (numSrcBoxTot);
  cp_dp = gpu_calloc_int (numSrcBoxTot);

  //Put data into the device
#ifdef DS_ORG
  gpu_copy_cpu2gpu_float (s_dp, P->src_, P->numSrc * 4);
  gpu_copy_cpu2gpu_float (t_dp, P->trg_, P->numTrg * 3);
#else
  gpu_copy_cpu2gpu_float(sx_dp, P->sx_, P->numSrc);
  gpu_copy_cpu2gpu_float(sy_dp, P->sy_, P->numSrc);
  gpu_copy_cpu2gpu_float(sz_dp, P->sz_, P->numSrc);
  gpu_copy_cpu2gpu_float(tx_dp, P->tx_, P->numTrg);
  gpu_copy_cpu2gpu_float(ty_dp, P->ty_, P->numTrg);
  gpu_copy_cpu2gpu_float(tz_dp, P->tz_, P->numTrg);
  gpu_copy_cpu2gpu_float(srcDen_dp, P->srcDen, P->numSrc);
#endif

  gpu_copy_cpu2gpu_int (tbdsf_dp, tbdsf, 2 * P->numTrgBox);
  gpu_copy_cpu2gpu_int (tbdsr_dp, tbdsr, 3 * numAugTrg);
  gpu_copy_cpu2gpu_int (cs_dp, cs, numSrcBoxTot);
  gpu_copy_cpu2gpu_int (cp_dp, cp, numSrcBoxTot);

  //kernel call
  int GRID_WIDTH=(int)ceil((float)numAugTrg/65535.0F);
  int GRID_HEIGHT=(int)ceil((float)numAugTrg/(float)GRID_WIDTH);
  dim3 BlockDim (BLOCK_HEIGHT,BLOCK_WIDTH);  //Block width will be 1
  dim3 GridDim (GRID_HEIGHT, GRID_WIDTH);    //Grid width should be 1
  //fprintf (stdout, "@@ [%s:%lu::p%d] numAugTrg=%d; BlockDim x GridDim = [%d x %d] x [%d x %d]\n", __FILE__, (unsigned long)__LINE__, mpirank, numAugTrg, BLOCK_HEIGHT, BLOCK_WIDTH, GRID_HEIGHT, GRID_WIDTH);

#ifdef DS_ORG
#if defined (__DEVICE_EMULATION__)
  GPU_MSG (">>> Device emulation mode <<<\n");
#endif
  if (numAugTrg) // No need to call kernel if numAugTrg == 0
    ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(t_dp,trgVal_dp,s_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg); GPU_CE;
#else
  ulist_kernel<<<GridDim,BLOCK_HEIGHT>>>(tx_dp,ty_dp,tz_dp,trgVal_dp,sx_dp,sy_dp,sz_dp,srcDen_dp,tbdsr_dp,tbdsf_dp,cs_dp,cp_dp,numAugTrg); GPU_CE;
#endif

  gpu_copy_gpu2cpu_float (P->trgVal, trgVal_dp, P->numTrg);

#ifdef DS_ORG
  cudaFree(s_dp); GPU_CE;
  cudaFree(t_dp); GPU_CE;
#else
  cudaFree(sx_dp);
  cudaFree(sy_dp);
  cudaFree(sz_dp);

  cudaFree(tx_dp);
  cudaFree(ty_dp);
  cudaFree(tz_dp);

  cudaFree(srcDen_dp);
#endif

  cudaFree(trgVal_dp); GPU_CE;
  cudaFree(tbdsf_dp); GPU_CE;
  cudaFree(tbdsr_dp); GPU_CE;
  cudaFree(cs_dp); GPU_CE;
  cudaFree(cp_dp); GPU_CE;

  free(cs);
  free(cp);
  free(tbdsf);
  free(tbdsr);
}
//}//end extern
