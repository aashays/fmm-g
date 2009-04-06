//#define SAVE_ME_FROM_FLTNUMMAT

//#define CERR
#define PI_4I 0.079577471F
//#define PI3_4I 0.238732413F
#include <cutil.h>
#include <cutil_inline.h>
#include "../p3d/upComp.h"
#include "../p3d/dnComp.h"
#include "../p3d/point3d.h"//dont remove this
#include "gpu_setup.h"

#include <iostream>

#define BLOCK_HEIGHT 64

__constant__ float3 sampos[320];	//undefined for everything greater than 295 for 6, greater than 191 for 4

__constant__ float3 samposDn[152];	//undefined for everything greater than 151 for 6 and 55 for 4

__global__ void up_kernel(float *src_dp,float *trgVal_dp,float *trgCtr_dp,float *trgRad_dp,int *srcBox_dp,int numSrcBox) {
	__shared__ float4 s_sh[BLOCK_HEIGHT];

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numSrcBox) {
		float3 trgCtr;
		float trgRad;
	//	float3 samp[5];
		float3 trg[5];
		float dX_reg;
		float dY_reg;
		float dZ_reg;
		int2 src=((int2*)srcBox_dp)[uniqueBlockId];	//x has start, y has size
		src.x+=threadIdx.x;

		trgCtr=((float3*)trgCtr_dp)[uniqueBlockId];
		trgRad=trgRad_dp[uniqueBlockId];

		//construct the trg

		trg[0].x=trgCtr.x+trgRad*sampos[4*threadIdx.x].x;
		trg[0].y=trgCtr.y+trgRad*sampos[4*threadIdx.x].y;
		trg[0].z=trgCtr.z+trgRad*sampos[4*threadIdx.x].z;
		trg[1].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+1].x;
		trg[1].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+1].y;
		trg[1].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+1].z;
		trg[2].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+2].x;
		trg[2].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+2].y;
		trg[2].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+2].z;
		trg[3].x=trgCtr.x+trgRad*sampos[4*threadIdx.x+3].x;
		trg[3].y=trgCtr.y+trgRad*sampos[4*threadIdx.x+3].y;
		trg[3].z=trgCtr.z+trgRad*sampos[4*threadIdx.x+3].z;
		trg[4].x=trgCtr.x+trgRad*sampos[256+threadIdx.x].x;
		trg[4].y=trgCtr.y+trgRad*sampos[256+threadIdx.x].y;
		trg[4].z=trgCtr.z+trgRad*sampos[256+threadIdx.x].z;

	//	int numSrc=srcBoxSize[uniqueBlockId];

		float4 tv=make_float4(0.0F,0.0F,0.0F,0.0F);
		float tve=0.0F;






		int num_chunk_loop=src.y/BLOCK_HEIGHT;
		for(int chunk=0;chunk<num_chunk_loop;chunk++) {
			__syncthreads();
			s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
			__syncthreads();

			src.x+=BLOCK_HEIGHT;

			for(int s=0;s<BLOCK_HEIGHT;s++) {
				dX_reg=s_sh[s].x-trg[0].x;
				dY_reg=s_sh[s].y-trg[0].y;
				dZ_reg=s_sh[s].z-trg[0].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.x+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[1].x;
				dY_reg=s_sh[s].y-trg[1].y;
				dZ_reg=s_sh[s].z-trg[1].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.y+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[2].x;
				dY_reg=s_sh[s].y-trg[2].y;
				dZ_reg=s_sh[s].z-trg[2].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.z+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[3].x;
				dY_reg=s_sh[s].y-trg[3].y;
				dZ_reg=s_sh[s].z-trg[3].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.w+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[4].x;
				dY_reg=s_sh[s].y-trg[4].y;
				dZ_reg=s_sh[s].z-trg[4].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tve+=dX_reg*s_sh[s].w;
				///////////////////////////////
			}

		}	//end num chunk loop
		__syncthreads();
		s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
		__syncthreads();
		for(int s=0;s<src.y%BLOCK_HEIGHT;s++) {
			dX_reg=s_sh[s].x-trg[0].x;
			dY_reg=s_sh[s].y-trg[0].y;
			dZ_reg=s_sh[s].z-trg[0].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.x+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[1].x;
			dY_reg=s_sh[s].y-trg[1].y;
			dZ_reg=s_sh[s].z-trg[1].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.y+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[2].x;
			dY_reg=s_sh[s].y-trg[2].y;
			dZ_reg=s_sh[s].z-trg[2].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.z+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[3].x;
			dY_reg=s_sh[s].y-trg[3].y;
			dZ_reg=s_sh[s].z-trg[3].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.w+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[4].x;
			dY_reg=s_sh[s].y-trg[4].y;
			dZ_reg=s_sh[s].z-trg[4].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tve+=dX_reg*s_sh[s].w;
			///////////////////////////////
		}	//end residual loop

		//write back
		tv.x*=PI_4I;
		tv.y*=PI_4I;
		tv.z*=PI_4I;
		tv.w*=PI_4I;
	//	tv.x=(float)trgCtr;
	//	tv.y=tv.z=tv.w=0.0F;
		((float4*)trgVal_dp)[uniqueBlockId*74+threadIdx.x]=tv;
		if(threadIdx.x<40)
			trgVal_dp[uniqueBlockId*296+256+threadIdx.x]=tve*PI_4I;
	}

}

__global__ void up_kernel_4(float *src_dp,float *trgVal_dp,float *trgCtr_dp,float *trgRad_dp,int *srcBox_dp,int numSrcBox) {
	__shared__ float4 s_sh[BLOCK_HEIGHT];

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numSrcBox) {
		float3 trgCtr;
		float trgRad;
	//	float3 samp[5];
		float3 trg[3];
		float dX_reg;
		float dY_reg;
		float dZ_reg;
		int2 src=((int2*)srcBox_dp)[uniqueBlockId];	//x has start, y has size
		src.x+=threadIdx.x;

		trgCtr=((float3*)trgCtr_dp)[uniqueBlockId];
		trgRad=trgRad_dp[uniqueBlockId];

		//construct the trg

		trg[0].x=trgCtr.x+trgRad*sampos[2*threadIdx.x].x;
		trg[0].y=trgCtr.y+trgRad*sampos[2*threadIdx.x].y;
		trg[0].z=trgCtr.z+trgRad*sampos[2*threadIdx.x].z;
		trg[1].x=trgCtr.x+trgRad*sampos[2*threadIdx.x+1].x;
		trg[1].y=trgCtr.y+trgRad*sampos[2*threadIdx.x+1].y;
		trg[1].z=trgCtr.z+trgRad*sampos[2*threadIdx.x+1].z;
		trg[2].x=trgCtr.x+trgRad*sampos[128+threadIdx.x].x;		//128 is blockheight*(trg2fetch-1)
		trg[2].y=trgCtr.y+trgRad*sampos[128+threadIdx.x].y;
		trg[2].z=trgCtr.z+trgRad*sampos[128+threadIdx.x].z;

	//	int numSrc=srcBoxSize[uniqueBlockId];

		float2 tv=make_float2(0.0F,0.0F);					//can be converted into a generic array.. not too big
		float tve=0.0F;






		int num_chunk_loop=src.y/BLOCK_HEIGHT;
		for(int chunk=0;chunk<num_chunk_loop;chunk++) {
			__syncthreads();
			s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
			__syncthreads();

			src.x+=BLOCK_HEIGHT;

			for(int s=0;s<BLOCK_HEIGHT;s++) {
				dX_reg=s_sh[s].x-trg[0].x;
				dY_reg=s_sh[s].y-trg[0].y;
				dZ_reg=s_sh[s].z-trg[0].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.x+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[1].x;
				dY_reg=s_sh[s].y-trg[1].y;
				dZ_reg=s_sh[s].z-trg[1].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tv.y+=dX_reg*s_sh[s].w;
				///////////////////////////////
				dX_reg=s_sh[s].x-trg[2].x;
				dY_reg=s_sh[s].y-trg[2].y;
				dZ_reg=s_sh[s].z-trg[2].z;

				dX_reg*=dX_reg;
				dY_reg*=dY_reg;
				dZ_reg*=dZ_reg;

				dX_reg += dY_reg+dZ_reg;

				dX_reg = rsqrtf(dX_reg);
				tve+=dX_reg*s_sh[s].w;
			}
		}	//end num chunk loop
		__syncthreads();
		s_sh[threadIdx.x]=((float4*)src_dp)[src.x];
		__syncthreads();
		for(int s=0;s<src.y%BLOCK_HEIGHT;s++) {
			dX_reg=s_sh[s].x-trg[0].x;
			dY_reg=s_sh[s].y-trg[0].y;
			dZ_reg=s_sh[s].z-trg[0].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.x+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[1].x;
			dY_reg=s_sh[s].y-trg[1].y;
			dZ_reg=s_sh[s].z-trg[1].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tv.y+=dX_reg*s_sh[s].w;
			///////////////////////////////
			dX_reg=s_sh[s].x-trg[2].x;
			dY_reg=s_sh[s].y-trg[2].y;
			dZ_reg=s_sh[s].z-trg[2].z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);
			tve+=dX_reg*s_sh[s].w;
			///////////////////////////////
		}	//end residual loop

		//write back
		tv.x*=PI_4I;
		tv.y*=PI_4I;
	//	tv.x=(float)trgCtr;
	//	tv.y=tv.z=tv.w=0.0F;
		((float2*)(trgVal_dp+uniqueBlockId*152))[threadIdx.x]=tv;	//in generic, float3 writes will be unrolled into multiple writes
		if(threadIdx.x<24)
			trgVal_dp[uniqueBlockId*152+128+threadIdx.x]=tve*PI_4I;
	}

}


void unmake_ds_up(float *trgValE,upComp_t *UpC) {
	int t=0;
	for(int i=0;i<UpC->numSrcBox;i++) {
		for(int j=0;j<UpC->trgDim;j++) {
//			assert(UpC->trgVal[i]!=NULL);

			if(UpC->trgVal[i]!=NULL)
				UpC->trgVal[i][j]=trgValE[t];
			t++;
//			cout<<i<<","<<j<<endl;
//			cout<<trgValE[t-1]<<endl;
		}
	}
}


void make_ds_up(int *srcBox,upComp_t *UpC) {	//TODO
	int start=0;
	int t=0;
	int size;
	for(int i=0;i<UpC->numSrcBox;i++) {
		srcBox[t++]=start;
		size=UpC->srcBoxSize[i];
		srcBox[t++]=size;
		start+=size;
	}
}

void gpu_up(upComp_t *UpC) {
	cout<<"Upward computation"<<endl;
#ifdef CERR
	cudaError_t C_E;
#endif
	cudaSetDevice(0);
//	unsigned int timer;
//	float ms;
//	cutCreateTimer(&timer);

	float *src_dp,*trgVal_dp,*trgCtr_dp,*trgRad_dp;
	int *srcBox_dp;

	float trgValE[UpC->trgDim*UpC->numSrcBox];
	int srcBox[2*UpC->numSrcBox];

	make_ds_up(srcBox,UpC);

	cudaMalloc((void**)&src_dp,sizeof(float)*UpC->numSrc * (UpC->dim+1)); CE(C_E)
	cudaMalloc((void**)&trgCtr_dp,sizeof(float)*UpC->numSrcBox*3); CE(C_E)
	cudaMalloc((void**)&trgRad_dp,sizeof(float)*UpC->numSrcBox); CE(C_E)
	cudaMalloc((void**)&srcBox_dp,sizeof(int)*UpC->numSrcBox*2); CE(C_E)
	cudaMalloc((void**)&trgVal_dp,sizeof(float)*UpC->trgDim*UpC->numSrcBox); CE(C_E)


	cudaMemcpy((void*)src_dp,UpC->src_,sizeof(float)*UpC->numSrc * (UpC->dim+1),cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)trgCtr_dp,UpC->trgCtr,sizeof(float)*UpC->numSrcBox*3,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)trgRad_dp,UpC->trgRad,sizeof(float)*UpC->numSrcBox,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)srcBox_dp,srcBox,sizeof(int)*UpC->numSrcBox*2,cudaMemcpyHostToDevice); CE(C_E)

	cudaMemcpyToSymbol(sampos,UpC->samPosF/*samp*/,sizeof(float)*UpC->trgDim*3); CE(C_E)
	int GRID_WIDTH=(int)ceil((float)UpC->numSrcBox/65535.0F);
	int GRID_HEIGHT=(int)ceil((float)UpC->numSrcBox/(float)GRID_WIDTH);
	dim3 GridDim(GRID_HEIGHT, GRID_WIDTH);
//	cout<<"Width: "<<GRID_WIDTH<<" HEIGHT: "<<GRID_HEIGHT<<endl;
	if(UpC->trgDim==296) {
		up_kernel<<<GridDim,BLOCK_HEIGHT>>>(src_dp,trgVal_dp,trgCtr_dp,trgRad_dp,srcBox_dp,UpC->numSrcBox); CE(C_E)
	}
	else if(UpC->trgDim==152) {
		up_kernel_4<<<GridDim,BLOCK_HEIGHT>>>(src_dp,trgVal_dp,trgCtr_dp,trgRad_dp,srcBox_dp,UpC->numSrcBox); CE(C_E)
	}
	else
		cout<<"Upward computations not implemented for this accuracy"<<endl;	//Exit the process?
		//also, a generic call can be put here


	cudaMemcpy(trgValE,trgVal_dp,sizeof(float)*UpC->trgDim*UpC->numSrcBox,cudaMemcpyDeviceToHost); CE(C_E)
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	ms = cutGetTimerValue(timer);
//	cout<<"Up kernel: "<<ms<<"ms"<<endl;
	unmake_ds_up(trgValE,UpC);	//FIXME: copies the gpu output into the 2d array used by the interface... make the interface use a 1d array

	cudaFree(src_dp);
	cudaFree(trgCtr_dp);
	cudaFree(trgRad_dp);
	cudaFree(srcBox_dp);
	cudaFree(trgVal_dp);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void make_ds_down(int *trgBox,dnComp_t *DnC) {
	int tt=0;
	int tot=0;
	for(int i=0;i<DnC->numTrgBox;i++) {
		int rem=DnC->trgBoxSize[i];
		while(rem>0) {
			trgBox[tt++]=tot;		//start
			int size=(rem<BLOCK_HEIGHT)?rem:BLOCK_HEIGHT;
			trgBox[tt++]=size;		//size
			trgBox[tt++]=i;			//box
			tot+=size;
			rem-=size;
		}
	}
}

void unmake_ds_down(float *trgValE,dnComp_t *DnC) {
	int t=0;
	for(int i=0;i<DnC->numTrgBox;i++) {
		for(int j=0;j<DnC->trgBoxSize[i];j++) {
			if(DnC->trgVal[i]!=NULL)
				DnC->trgVal[i][j]=trgValE[t++];
		}
	}
}

__global__ void dn_kernel(float *trg_dp,float *trgVal_dp,float *srcCtr_dp,float *srcRad_dp,int *trgBox_dp,float *srcDen_dp,int numAugTrg) {
	__shared__ float4 s_sh[64];
	int3 trgBox;

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numAugTrg) {
		trgBox=((int3*)trgBox_dp)[uniqueBlockId];		//start,size,box

		float3 t_reg=((float3*)trg_dp)[trgBox.x+threadIdx.x];

		float3 srcCtr=((float3*)srcCtr_dp)[trgBox.z];
		float srcRad=srcRad_dp[trgBox.z];

		float dX_reg,dY_reg,dZ_reg;
		float tv_reg=0.0;

		//every thread computes a single src body


		s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[threadIdx.x].x;
		s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[threadIdx.x].y;
		s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[threadIdx.x].z;

		s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x];

		__syncthreads();
		for(int src=0;src<64;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);

			tv_reg+=dX_reg*s_sh[src].w;
		}
		__syncthreads();
		s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[64+threadIdx.x].x;
		s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[64+threadIdx.x].y;
		s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[64+threadIdx.x].z;

		s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x+64];

		__syncthreads();
		for(int src=0;src<64;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);

			tv_reg+=dX_reg*s_sh[src].w;
		}
		__syncthreads();
		if(threadIdx.x<24) {
			s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[128+threadIdx.x].x;
			s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[128+threadIdx.x].y;
			s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[128+threadIdx.x].z;

			s_sh[threadIdx.x].w=srcDen_dp[152*trgBox.z+threadIdx.x+128];
		}

		__syncthreads();
		for(int src=0;src<24;src++) {
			dX_reg=s_sh[src].x-t_reg.x;
			dY_reg=s_sh[src].y-t_reg.y;
			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);

			tv_reg+=dX_reg*s_sh[src].w;
		}

		if(threadIdx.x<trgBox.y)
			trgVal_dp[trgBox.x+threadIdx.x]=tv_reg*PI_4I;
//			trgVal_dp[trgBox.x+threadIdx.x]=trgBox.z;
	}//extra padding block

}

__global__ void dn_kernel_4(float *trg_dp,float *trgVal_dp,float *srcCtr_dp,float *srcRad_dp,int *trgBox_dp,float* srcDen_dp,int numAugTrg) {

	__shared__ float4 s_sh[56];
	int3 trgBox;

	int uniqueBlockId=blockIdx.y * gridDim.x + blockIdx.x;
	if(uniqueBlockId<numAugTrg) {
		trgBox=((int3*)trgBox_dp)[uniqueBlockId];		//start,size,box

		float3 t_reg=((float3*)trg_dp)[trgBox.x+threadIdx.x];

		float3 srcCtr=((float3*)srcCtr_dp)[trgBox.z];
		float srcRad=srcRad_dp[trgBox.z];

		float dX_reg,dY_reg,dZ_reg;
		float tv_reg=0.0;

		//every thread computes a single src body

		if(threadIdx.x<56) {	//no segfaults here

			s_sh[threadIdx.x].x=srcCtr.x+srcRad*samposDn[threadIdx.x].x;
			s_sh[threadIdx.x].y=srcCtr.y+srcRad*samposDn[threadIdx.x].y;
			s_sh[threadIdx.x].z=srcCtr.z+srcRad*samposDn[threadIdx.x].z;

			s_sh[threadIdx.x].w=srcDen_dp[56*trgBox.z+threadIdx.x];
		}
		__syncthreads();
		for(int src=0;src<56;src++) {
			dX_reg=s_sh[src].x-t_reg.x;

			dY_reg=s_sh[src].y-t_reg.y;

			dZ_reg=s_sh[src].z-t_reg.z;

			dX_reg*=dX_reg;
			dY_reg*=dY_reg;
			dZ_reg*=dZ_reg;

			dX_reg += dY_reg+dZ_reg;

			dX_reg = rsqrtf(dX_reg);

			tv_reg+=dX_reg*s_sh[src].w;
		}

		if(threadIdx.x<trgBox.y)
			trgVal_dp[trgBox.x+threadIdx.x]=tv_reg*PI_4I;
	}//extra padding block

}

int getnumAugTrg(dnComp_t *DnC) {
	int numAugTrg=0;
	for(int i=0;i<DnC->numTrgBox;i++) {
		numAugTrg+=(int)ceil((float)DnC->trgBoxSize[i]/(float)BLOCK_HEIGHT);
	}
	return numAugTrg;
}

void gpu_down(dnComp_t *DnC) {
#ifdef CERR
	cudaError_t C_E;
#endif
	cudaSetDevice(0);
//	DnC->numTrgBox=75;
	float *trg_dp,*trgVal_dp,*srcCtr_dp,*srcRad_dp,*srcDen_dp;
//	int *srcBoxSize_dp,srcBoxStart_dp;
	int *trgBox_dp;	//has start and size and block
//	float trgValE[DnC->numTrg];
	float *trgValE=(float*)calloc(DnC->numTrg,sizeof(float));
	if(trgValE==NULL) cout<<"segfault imminent"<<endl;
	int numAugTrg=getnumAugTrg(DnC);
	int trgBox[3*numAugTrg];
//	cout<<"srcdim is: "<<DnC->srcDim<<endl;
	make_ds_down(trgBox,DnC);

//	cout<<"srcctr: "<<endl;
//	for(int i=0;i<3*DnC->numTrgBox;i++) {
//		cout<<DnC->srcCtr[i]<<endl;
//	}

	cudaMalloc((void**)&trg_dp,sizeof(float)*(DnC->numTrg+BLOCK_HEIGHT) * (DnC->dim)); CE(C_E)
	cudaMalloc((void**)&srcCtr_dp,sizeof(float)*DnC->numTrgBox*3); CE(C_E)
	cudaMalloc((void**)&srcRad_dp,sizeof(float)*DnC->numTrgBox); CE(C_E)
	cudaMalloc((void**)&trgBox_dp,sizeof(int)*numAugTrg*3); CE(C_E)
	cudaMalloc((void**)&trgVal_dp,sizeof(float)*DnC->numTrg); CE(C_E)
	cudaMalloc((void**)&srcDen_dp,sizeof(float)*DnC->numTrgBox*DnC->srcDim); CE(C_E)

	cudaMemcpy((void*)trg_dp,DnC->trg_,sizeof(float)*DnC->numTrg * (DnC->dim),cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)srcCtr_dp,DnC->srcCtr,sizeof(float)*DnC->numTrgBox*3,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)srcRad_dp,DnC->srcRad,sizeof(float)*DnC->numTrgBox,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)trgBox_dp,trgBox,sizeof(int)*numAugTrg*3,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpy((void*)srcDen_dp,DnC->srcDen,sizeof(float)*DnC->numTrgBox*DnC->srcDim,cudaMemcpyHostToDevice); CE(C_E)
	cudaMemcpyToSymbol(samposDn,DnC->samPosF,sizeof(float)*DnC->srcDim*3); CE(C_E)
//	int GRID_HEIGHT=UpC->numSrcBox;
	int GRID_WIDTH=(int)ceil((float)numAugTrg/65535.0F);
	int GRID_HEIGHT=(int)ceil((float)numAugTrg/(float)GRID_WIDTH);
	dim3 GridDim(GRID_HEIGHT, GRID_WIDTH);
//	cout<<"Width: "<<GRID_WIDTH<<" HEIGHT: "<<GRID_HEIGHT<<endl;
	if(DnC->srcDim==152) {
		dn_kernel<<<GridDim,BLOCK_HEIGHT>>>(trg_dp,trgVal_dp,srcCtr_dp,srcRad_dp,trgBox_dp,srcDen_dp,numAugTrg); CE(C_E)
	}
	else if(DnC->srcDim==56) {
		dn_kernel_4<<<GridDim,BLOCK_HEIGHT>>>(trg_dp,trgVal_dp,srcCtr_dp,srcRad_dp,trgBox_dp,srcDen_dp,numAugTrg); CE(C_E)
	}
	else
		cout<<"Downward computations not implemented for this accuracy"<<endl;	//Exit the process?
		//also, a generic call can be put here


	cudaMemcpy(trgValE,trgVal_dp,sizeof(float)*DnC->numTrg,cudaMemcpyDeviceToHost); CE(C_E)
	unmake_ds_down(trgValE,DnC);	//FIXME: copies the gpu output into the 2d array used by the interface... make the interface use a 1d array


	cudaFree(trg_dp);
	cudaFree(srcCtr_dp);
	cudaFree(srcRad_dp);
	cudaFree(trgBox_dp);
	cudaFree(trgVal_dp);
	cudaFree(srcDen_dp);
}
