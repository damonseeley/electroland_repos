#ifndef __POINT_CLOUD_CU__
#define __POINT_CLOUD_CU__

#include <math.h>
#include <cutil_inline.h>
#include "GestureTypeDefs.h"

//#define EMULATE

#define MAX_THREADS_PER_BLOCK 512
// this const should be defined in CUDA libs but I can't find it



__global__ void gpu_calcPointCloud_kernel(int camCnt, int imgWidth, int imgHeight, double* params, double* transforms, unsigned short *imgs, float* results) 
{

	int u,v,t,pixelIndex;

	t = threadIdx.z; 
	t = (t >= camCnt) ? -1: t;

	u = (blockIdx.x * blockDim.x) + threadIdx.x;
	u = (u >= imgWidth) ? -1 : u;

	v = (blockIdx.y * blockDim.y) + threadIdx.y;
	v = (v >= imgHeight) ? -1 : v;

	
	bool isValidIndex = (u >= 0) && (v >= 0) && (t>=0);


#ifdef EMULATE
	if((u == 0) &&(v == 0)) {
		printf("block %i %i %i, thead %i %i %i --> %i %i %i \n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, u,v,t);
		}
#endif
//	printf("isValidPoint %i \n", isValidPoint );


	pixelIndex = isValidIndex ? ( (t * imgWidth * imgHeight) + (v * imgWidth) + u ) : -1; 
	unsigned short z = isValidIndex ? imgs[pixelIndex] : 0;	
	

	// avoid branching by zeroing out if u or v is out of bounds
	z = (z == USHRT_MAX) ? 0 : z; // if z is not valid 
	
	//	z = (z < 1000) ? -10000 : z;

	isValidIndex = (z != 0) && isValidIndex;

	
	double* paramPt = &params[t * CAM_PARAM_CNT];
	double cx = *paramPt++;
	double cy = *paramPt++;
	double cu = *paramPt++;
	double cv = *paramPt++;	
	
	#ifdef EMULATE
	if((u == 0) &&(v == 0)) {
	printf("cx %f,  cy %f,    cu %f,  cv %f \n", cx,cy,cu,cv);
	}
	#endif

	double x = isValidIndex ?  ((u - cu) *  cx * (double) z) : 0.0;
	double y = isValidIndex ?  ((v - cv) *  cy * (double) z) : 0.0;
	

// if point or pixelIndex is invalid use 0 as pixel index
	pixelIndex = isValidIndex ? pixelIndex++ : 0 ;
	pixelIndex *= 3;

	


		
//	double *m = &transforms[t * 16];
//	double tx =	*(m)   * x + *(m+4) * y + *(m+8)  * z + *(m+12);
//	double ty =	*(m+1) * x + *(m+5) * y + *(m+9)  * z + *(m+13);
//	double tz =	*(m+2) * x + *(m+6) * y + *(m+10) * z + *(m+14);
//	double w =	*(m+3) * x + *(m+7) * y + *(m+11) * z + *(m+15);
	// we should be doing affine transforms but lets devide by w anyway
//	w = 1.0/w; // invert w

//	results[pixelIndex++]	=  (float) (tx * w); 
//	results[pixelIndex++]	=  (float) (ty * w) ; 
//	results[pixelIndex]		=  (float) (tz * w) ;
double *m = &transforms[t*12];
double tx =  m[0] * x	+  m[1] * y	+ m[2] * z + m[9];	
double ty =  m[3] * x	+  m[4] * y	+ m[5] * z + m[10];	
double tz =  m[6] * x	+  m[7] * y	+ m[8] * z + m[11];
results[pixelIndex++]	=  isValidIndex ? (float) tx : 0.0f ;
results[pixelIndex++]	=  isValidIndex ? (float) ty : 0.0f ; 
results[pixelIndex]		=  isValidIndex ? (float) tz : 0.0f ;

		
	}


	// Wrapper for the __global__ call that sets up the kernel call
	extern "C" void gpu_calcPointCloud(int camCnt, int imgWidth, int imgHeight, double* params, double* transforms, unsigned short *imgs, float* results)
	{
		


		dim3 threadDims(16,8,camCnt);
		dim3 blockDims((int) ceilf(imgWidth/ threadDims.x), (int) ceilf(imgHeight/threadDims.y));
		//printf("threadDims: (%i, %i, %i)   blockDims: (%i, %i, %i) \n", threadDims.x, threadDims.y, threadDims.z, blockDims.x, blockDims.y, blockDims.z);
		gpu_calcPointCloud_kernel <<< blockDims, threadDims>>> (camCnt, imgWidth, imgHeight, params, transforms, imgs, results);



	  
	}

	#endif