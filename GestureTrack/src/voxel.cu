#ifndef __VOXEL_CU__
#define __VOXEL_CU__

#include <math.h>
#include <cutil_inline.h>


#define MAX_THREADS_PER_BLOCK 512

//#define EMULATE 

#ifdef EMULATE
#define ATOMIC_ADD(a,v) *a+v
#else
#define ATOMIC_ADD(a,v) atomicAdd(a,v)
#endif

// this const should be defined in CUDA libs but I can't find it

__device__ int getIndex(float val, float minDim, float maxDim, int divs) {
	float divSize = (maxDim - minDim)/divs;
	int result =  (int) (val-minDim)/divSize; // will be negative if less than bounds
	result = (result < divs) ? result : -1; // if greater than upper bound set negative to denote out of bounds
	return result;
} 

__global__ void gpu_calcVoxel_kernel(int pointCnt, float* pixels, float minX, float minY, float minZ, float maxX, float maxY, float maxZ, int divX, int divY, int divZ, float* voxGrid) 
{

#ifdef EMULATE
	printf("------\n");
#endif
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	bool validPoint = ((i > 0) && (i < pointCnt))? true : false;
#ifdef EMULATE	
	printf("%i is a is valid? %i\n", i, validPoint);
#endif

	i *=3; // each point is x,y,z
	
	float px = validPoint ? pixels[i++] : -1;
	float py = validPoint ? pixels[i++] : -1;
	float pz = validPoint ? pixels[i] : -1;
	
#ifdef EMULATE	
		printf("    pt (%f,%f, %f)\n", px,py,pz);
#endif
	
	int vx = getIndex(px, minX, maxX, divX);
	int vy = getIndex(py, minY, maxY, divY);
	int vz = getIndex(pz, minZ, maxZ, divZ);
#ifdef EMULATE	
			printf("   i (%i,%i, %i)\n", vx,vy,vz);
#endif
	
	validPoint = validPoint && (vx >= 0) && (vy >= 0) && (vz >= 0) && (vz > 0);
	
	// if not valid pic a random voxel to "inc" by zero to avoid deadlocking during atomic add
	// not sure if this is really neeeded but it can't hurt
	int vi = validPoint ? (vz * divX * divY) + (vy * divY) + vx : i % (divX*divY*divY);
	float inc = validPoint ? 1.0f : 0.0f;
	ATOMIC_ADD(&voxGrid[vi], inc);	
	
	
}

// we want the order to be back to front, bottom to top
// so we want 0, divy - y, 0 to be first

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void gpu_calcVoxel(int pointCnt, float* pixels, float minX, float minY, float minZ, float maxX, float maxY, float maxZ, int divX, int divY, int divZ, float* voxGrid)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(pointCnt/(float) theadsPerBlock);
	gpu_calcVoxel_kernel <<<blocks,theadsPerBlock>>> (pointCnt, pixels, minX, minY, minZ, maxX, maxY, maxZ, divX, divY, divZ, voxGrid);

};









__global__ void gpu_addScale_kernel(int gridSize, float* d_this, float a, float* d_that, float b)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	#ifdef EMULATE
		printf("%i %i %f %f\n", i, gridSize, a, b);
	#endif
	
	#ifdef EMULATE
		if(i< gridSize) {
			printf("this:%f    that:%f ----> %f\n", d_this[i], d_that[i], ((d_this[i] * a) + (d_that[i] * b)));
						
		}
	#endif
	
	
	if(i < gridSize) 
		d_this[i] = ((d_this[i] * a) + (d_that[i] * b));
		
};


	extern "C" void gpu_addScale(int gridSize, float* d_this, float a, float* d_that, float b)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_addScale_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this,  a,  d_that,  b);

};

__global__ void gpu_mult_kernel(int gridSize, float* d_this, float* d_that)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		
	if(i < gridSize) 
		d_this[i] *= d_that[i];
		
};


extern "C" void gpu_mult(int gridSize, float* d_this, float* d_that)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_mult_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, d_that);

};


__global__ void gpu_add_kernel(int gridSize, float* d_this, float* d_that)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		
	if(i < gridSize) 
		d_this[i] += d_that[i];
		
};


extern "C" void gpu_add(int gridSize, float* d_this, float* d_that)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_add_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, d_that);

};

__global__ void gpu_sub_kernel(int gridSize, float* d_this, float* d_that)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		
	if(i < gridSize) 
		d_this[i] -= d_that[i];
		
};

__global__ void gpu_sub_kernel2(int gridSize, float* d_this, float* d_a, float* d_b)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		
	if(i < gridSize) 
		d_this[i] = d_a[i] - d_b[i];
		
};

extern "C" void gpu_sub(int gridSize, float* d_this, float* d_that)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_sub_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, d_that);

};

extern "C" void gpu_sub2(int gridSize, float* d_this, float* d_a, float* d_b)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_sub_kernel2 <<<blocks,theadsPerBlock>>> ( gridSize, d_this, d_a, d_b);

};


__global__ void gpu_thresh_kernel(int gridSize, float* d_this, float thresh)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(i < gridSize) 
		d_this[i] = d_this[i] >= thresh ? d_this[i]: 0.0f	;	
};

extern "C" void gpu_thresh(int gridSize, float* d_this, float thresh)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_thresh_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, thresh);

};

__global__ void gpu_threshSet_kernel(int gridSize, float* d_this, float thresh, float belowVal, float aboveVal)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(i < gridSize) 
		d_this[i] = d_this[i] >= thresh ? aboveVal : belowVal;	
};

extern "C" void gpu_threshSet(int gridSize, float* d_this, float thresh, float belowVal, float aboveVal)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_threshSet_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, thresh, belowVal, aboveVal);

};





__global__ void gpu_scalarMult_kernel(int gridSize, float* d_this, float val)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(i < gridSize) 
		d_this[i] *= val;	
		
};

extern "C" void gpu_scalarMult(int gridSize, float* d_this, float val)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_thresh_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, val);

};

__global__ void gpu_incIfOverThresh_kernel(int gridSize, float *d_this, float* d_that, float thresh)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(i < gridSize) 
		d_this[i] = (d_that[i] > thresh) ? d_this[i]+1 : 0;	
	
};


extern "C" void gpu_incIfOverThresh(int gridSize, float *d_this, float* d_that, float thresh){
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(gridSize/(float) theadsPerBlock);
	gpu_incIfOverThresh_kernel <<<blocks,theadsPerBlock>>> ( gridSize, d_this, d_that, thresh);

};




#endif

