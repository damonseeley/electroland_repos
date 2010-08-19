#ifndef __PROJECTION_CU__
#define __PROJECTION_CU__

#include "ProjCell.h"

#include <math.h>
#include <cutil_inline.h>

//#define EMULATE

#ifdef EMULATE
	#define ATOMIC_ADD(a,v) *a+v
	#define ATOMIC_MIN(a,v) min(*a,v)
	#define ATOMIC_MAX(a,v) max(*a,v)
#else
	#define ATOMIC_ADD(a,v) atomicAdd(a,v)
	#define ATOMIC_MIN(a,v) atomicMin(a,v)
	#define ATOMIC_MAX(a,v) atomicMax(a,v)
#endif

__global__ void gpu_calcProjection_kernel(float* d_voxGrid, int voxSize, int dx, int dy, int dz, float thresh, ProjCell* d_cells) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		
		
	int z = i / (dx * dy);
	int r = i % (dx * dy);
	int y = r / (dx);
	int x = r % (dx);


	if(((i < voxSize) && (x < dx -1)) && ((y < dy -1) && (z < dz - 1))) {
		float val = d_voxGrid[i];
		if(val > thresh) {
			ProjCell *cell = &d_cells[x + (z * dx)];
			ATOMIC_MIN(&cell->min, y);	
			ATOMIC_MAX(&cell->max, y);	
			ATOMIC_MIN(&cell->min, y);	
			ATOMIC_ADD(&cell->total, val);
			ATOMIC_ADD(&cell->cnt,1);
		}		

		//int vi = validPoint ? (vz * divX * divY) + (vy * divY) + vx : i % (divX*divY*divY);
		//float inc = validPoint ? 1.0f : 0.0f;
		//ATOMIC_ADD(&voxGrid[vi], inc);	

		
	}
		
};

__global__ void gpu_conv_kernel(float* d_src, float* d_dst, int w, int h, float *d_conv, int cWidth, int cHeight, bool mirrorBoarder) {
	int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if( ind < (w*h)) {
		int y = ind /  w;
		int x = ind % w;
		
		float origCellVal = d_src[x + y*w];
		float sum = 0;
		
		int halfWidth = cWidth/2;
		int halfHeight = cHeight/2;
		int convX = 0;
		for(int i  = -halfWidth; i <= halfWidth; i++) {
			int convY = 0;
			for(int j  = -halfHeight; j <= halfHeight; j++) {
				int xOff = x + i;
				int yOff = y + j;
				float cellVal = 0;
				if(((xOff<0) || (yOff<0))||((xOff>=w)||(yOff>=h))) {
					if(mirrorBoarder)		
						cellVal = origCellVal;
				} else {
					cellVal = d_src[xOff +  (yOff * w)];
				}
				sum+= cellVal * d_conv[convX + (convY * cWidth)];
				convY++;	
			}
			convX++;
		}
		d_dst[x+y*w] = sum;
	}
	

}
extern "C" void gpu_calcProjection(float* d_voxGrid, int dx, int dy, int dz, float thresh, ProjCell* d_cells) {
	int threadsPerBlock = 256;
	int voxSize = dx*dy*dz;
	int blocks = (int) ceilf(voxSize/(float) threadsPerBlock);
	gpu_calcProjection_kernel <<<blocks,threadsPerBlock>>> (d_voxGrid, voxSize, dx, dy, dz, thresh, d_cells);
}


extern "C" void gpu_conv(float* d_src, float* d_dst, int w, int h, float *d_conv, int cWidth, int cHeight, bool mirrorBoarder) {
	int threadsPerBlock = 256;
	int voxSize = w*h;
	int blocks = (int) ceilf(voxSize/(float) threadsPerBlock);
	gpu_conv_kernel<<<blocks, threadsPerBlock>>>(d_src, d_dst, w, h, d_conv, cWidth, cHeight, mirrorBoarder);

}


#endif