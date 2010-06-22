#ifndef __POINT_CLOUD_CU__
#define __POINT_CLOUD_CU__

#include <cutil_inline.h>


#define MAX_THREADS_PER_BLOCK 512
// this const should be defined in CUDA libs but I can't find it

__global__ void gpu_calcPointCloud_kernel(int imgWidth, int imgHeight, double* params, unsigned short *imgs, float* results) 
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	int t = threadIdx.z; // image (or camera) number

	int pixelIndex = (t * imgWidth * imgHeight) + (v * imgWidth) + u;
	float z = (float) imgs[pixelIndex]; // get z from image 1 or 2
	

	// avoid branching by zeroing out if u or v is out of bounds
	z = (u > imgWidth) ? 0 : z;
	z = (v > imgHeight) ? 0 : z;

	z = (z == 65535) ? 0 : z; // if z is not valid 
	
	
	int resultIndex = pixelIndex * 3;
	
	double *ptParam = &params[t];
	
	double cx = *ptParam++;
	double cy = *ptParam++;
	double centerU = *ptParam++;
	double centerV = *ptParam;
	
	results[resultIndex++] = ((float)((u - centerU) * cx)) * z; 
	results[resultIndex++]  =((float)((v - centerV) * cy)) * z;
	results[resultIndex] = (float)z;
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void gpu_calcPointCloud(int camCnt, int imgWidth, int imgHeight, double* params, unsigned short *imgs, float* results)
{
	// blocks
	// common to have 256 theads per block
	
	
	int totalPoints = imgWidth * imgHeight;
	int gridSize =  (int) sqrt((float) MAX_THREADS_PER_BLOCK / (float) camCnt);

	dim3 threadsPerBlock(16,16,camCnt);
	dim3 numBlocks(totalPoints / threadsPerBlock.x, totalPoints / threadsPerBlock.y);

    gpu_calcPointCloud_kernel<<< numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, params, imgs, results );
}

#endif