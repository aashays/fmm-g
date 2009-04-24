#include "gpu_vlist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cutil.h>

#include <sys/time.h>
#include <assert.h>
#include "cudacheck.h"
#include "gpu_setup.h"

using namespace std;

#define BLOCK_X_SIZE 1
#define BLOCK_Y_SIZE 64


const int numSources=1024*8;
const int numTargets=1024*8;



const int NL= 320; //length of density and potential vectors per box
const int NS= 210; //number of symmetries
typedef float2 Field[NL];
typedef int Index[NL];

struct timeval  tp;
struct timezone tzp;



#define get_seconds()   (gettimeofday(&tp, &tzp), \
                        (double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)



//These parameters used for random generator
const float maxRandomNumber=31.0f;
const float n=pow(2,maxRandomNumber) -1;
int a = 48271;
int q = 44488;
int r = 3399;
float temp=1;

float ran=1;

float Lehmer(float x){
        float t;
        t = a*((int) x%q) - r* ((int) x/q);
        if (t > 0) return t;
        else return (t+n);
}


float RandomGenerator(){
    temp=Lehmer(temp);
    return (temp/n);
}

void vlistGenerators(int *vlistTemp, int vlistSize){
      for (int i=0;i<vlistSize;i++){
 //           vlist[i][j]= (int( RandomGenerator()*numSources*100))%numSources;
//            assert(vlist[i][j]<numSources);
            vlistTemp[i]= (int( RandomGenerator()*numSources*100))%numSources;
 //vlist[i][j];
      }

}





void symmetryGenerators(int *symTemp, int vlistSize){
      int random;
         symTemp[0]=0;
         for (int i=1;i<vlistSize;i++){
           random=(int(RandomGenerator()*NS))%(NS);
            symTemp[i]= random;
         }
}


void densityGenerators(Field d[numSources]){
   for (int i=0;i<numSources;i++){
      for (int j=0;j<NL;j++){
         d[i][j].x=RandomGenerator();
         d[i][j].y=RandomGenerator();

      }
   }
}

void translationGenerators(Field T[NS]){
   for (int i=0;i<NS;i++){
      for (int j=0;j<NL;j++){
         T[i][j].x=RandomGenerator();
         T[i][j].y=RandomGenerator();

      }
   }
}


void vlistStartGenerators(int vlistStart[numTargets]){
   vlistStart[0]=1;
   for (int i=1;i<numTargets+1;i++){
      vlistStart[i]=vlistStart[i-1]+(int (RandomGenerator()*(NS)*100))%(NS);
   }

}


__global__ void Potential(int NumSym, int DenLength,float2 *T, float2 *D,int *vlist,int *vlistStart,int *spv,float2 *pCopy){


   int j,indexVlist,indexSpv;
   int tx =threadIdx.x;
   int txid=blockIdx.x*blockDim.x + tx;
   int tyid = blockIdx.y*blockDim.y +threadIdx.y;

   float2 Tl,dl;
//   int Il;
   float2 pot;
   
  pot.x=0;
   pot.y=0;

   for (j=vlistStart[txid];j<vlistStart[txid+1];j++){
      indexVlist = vlist[j];
      indexSpv = spv[j];

      dl=D[indexVlist*DenLength+tyid];
      Tl=T[indexSpv*DenLength+tyid];

       pot.x+=Tl.x*dl.x;
       pot.x-=Tl.y*dl.y;

       pot.y+=Tl.x*dl.y;

       pot.y+=Tl.y*dl.x;


   }
   
     pCopy[txid*DenLength+tyid]=pot;
   }



//#define BLOCK_X_SIZE 1
//#define BLOCK_Y_SIZE 80
double CUDAexecTime;

void cudavlistfunc(int DenLength,int NumSym,int numTargets,int numSources,int *vlistStart,int *vlist,int *spv,float *d,float *T,float *p){

  GPU_MSG ("V-List");
  if (!numTargets) return;

  int* cudaVlist,*cudaSpv,*cudaVlistStart;

   float2* cudaD,*cudaP,*cudaT;

typedef float2 Field1[DenLength];
typedef int Index1[DenLength];


   int sizeField = sizeof(Field1);

   if (DenLength&(BLOCK_Y_SIZE-1)!=0) {
      printf("The density length should be the multiples of BLOCK_Y_SIZE\n");
      return;
   }

   int vlistSize=vlistStart[numTargets];

//printf("Density at point[1].y is %f\n",d[1].y);
   //cudaMalloc((void**) &cudaVlist, vlistSize*sizeof(int));
   //cudaMalloc((void**) &cudaSpv,  vlistSize*sizeof(int));
   cudaVlist = gpu_calloc_int (vlistSize);
   cudaSpv = gpu_calloc_int (vlistSize);

   //cudaMalloc((void**) &cudaD,numSources*sizeField);
   //cudaMalloc((void**) &cudaP,numTargets*sizeField);
   //cudaMalloc((void**)& cudaT,NumSym*sizeField);
   //cudaMalloc((void**)&cudaVlistStart,(numTargets+1)*sizeof(int));

   cudaD = (float2 *)gpu_calloc (numSources*sizeField);
   cudaP = (float2 *)gpu_calloc (numTargets*sizeField);
   cudaT = (float2 *)gpu_calloc (NumSym*sizeField);
   cudaVlistStart = gpu_calloc_int (numTargets+1);

   //cudaMemcpy(cudaVlist,vlist, vlistSize*sizeof(int),cudaMemcpyHostToDevice);
   //cudaMemcpy(cudaSpv,spv,vlistSize*sizeof(int),cudaMemcpyHostToDevice);
   //cudaMemcpy(cudaD,d,numSources*sizeField,cudaMemcpyHostToDevice);
   //cudaMemcpy(cudaT,T,NumSym*sizeField,cudaMemcpyHostToDevice);
   //cudaMemcpy(cudaVlistStart,vlistStart,(numTargets+1)*sizeof(int),cudaMemcpyHostToDevice);

   gpu_copy_cpu2gpu_int (cudaVlist, vlist, vlistSize);
   gpu_copy_cpu2gpu_int (cudaSpv, spv, vlistSize);
   gpu_copy_cpu2gpu (cudaD, d, numSources*sizeField);
   gpu_copy_cpu2gpu (cudaT, T, NumSym*sizeField);
   gpu_copy_cpu2gpu_int (cudaVlistStart, vlistStart, numTargets+1);

    checkCUDAError("memcpy");

   dim3 dimBlock(BLOCK_X_SIZE,BLOCK_Y_SIZE);
   dim3 dimGrid(numTargets/BLOCK_X_SIZE,DenLength/BLOCK_Y_SIZE);


   Potential<<<dimGrid,dimBlock>>>(NumSym,DenLength,cudaT,cudaD,cudaVlist,cudaVlistStart,cudaSpv,cudaP);

   //cudaMemcpy(p,cudaP,numTargets*sizeField,cudaMemcpyDeviceToHost);
   gpu_copy_gpu2cpu (p, cudaP, numTargets*sizeField);
   checkCUDAError("memcpy");

   cudaFree(cudaVlist);
   cudaFree(cudaSpv);
   cudaFree(cudaT);
   cudaFree(cudaD);
   cudaFree(cudaP);
}



void vlistfunc(int NL,int NS,int numTargets,int numSources,int *vlistSize,int *vlist,int *spv,float d[numSources][NL][2],float T[NS][NL][2],float p[numTargets][NL][2]){

   int i,j;
   float2 df[numSources*NL],Tf[NS*NL],pf[numTargets*NL];
//convert from float[2] to float2
   for (i=0;i<numSources;i++){
      for (j=0;j<NL;j++){
         df[i*NL+j].x=  d[i][j][0];
         df[i*NL+j].y =  d[i][j][1];
      }
   }


   for (i=0;i<NS;i++){
      for (j=0;j<NL;j++){
         Tf[i*NL+j].x =  T[i][j][0];
         Tf[i*NL+j].y =  T[i][j][1];
      }
   }


// cudavlistfunc(NL,NS,numTargets,numSources,vlistSize,vlist,spv,df,Tf,pf);




  for (i=0;i<numTargets;i++){
      for (j=0;j<NL;j++){
         p[i][j][0]=  pf[i*NL+j].x;
         p[i][j][1] =  pf[i*NL+j].y;
      }
   }



}

