#include "Projection.h"

#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
#include <cutil_inline.h>

extern "C" void gpu_calcProjection(float* d_voxGrid, int dx, int dy, int dz, float thresh, ProjCell* d_grid);
extern "C" void gpu_conv(float* d_src, float* d_dst, int w, int h, float *d_conv, int cWidth, int cHeight, bool mirrorBorder);

Projection::Projection(Vec3i div) {
	this->div = div;
	grid = new ProjCell[div.x * div.z];
	if(grid == NULL) {
		std::cerr << "Projection cannot allocate memory for grid of size " << div.x << "x" << div.z << std::endl;
		exit(1);
	}
	gridMemSize = sizeof(ProjCell) * div.x * div.z;
	d_grid = NULL;
}


Projection::~Projection() {
	delete grid;
}


void Projection::allocateGridOnGPU(bool copy) { 
	if(d_grid == NULL) { // if not null already on device
		cutilSafeCall( cudaMalloc((void**)&d_grid, gridMemSize) ); // need to look up safecall TODO
		if(copy) {
				cutilSafeCall( cudaMemcpy(d_grid, grid, gridMemSize, cudaMemcpyHostToDevice) );
		} else {
				cudaMemset(d_grid, 0, gridMemSize);
		}
	}
}

void Projection::deallocateGridOnGPU() {
		cudaFree(d_grid);
		d_grid = NULL;
}

void Projection::calcProjection(Voxel *voxel, float thresh, bool freeGridFromGPU) {	
	float *d_voxGrid = voxel->getGridOnGPU();

	allocateGridOnGPU(false);


	gpu_calcProjection(d_voxGrid, voxel->divisions.x, voxel->divisions.y, voxel->divisions.z, thresh, d_grid);

	cutilSafeCall( cudaMemcpy(grid, d_grid, gridMemSize, cudaMemcpyDeviceToHost) );

	if(freeGridFromGPU)
		deallocateGridOnGPU();

}

void Projection::setRenderLocation(Vec3f min, Vec3f max) { 
	minLoc = min; 
	maxLoc = max; 
	cellRenderSize = maxLoc - minLoc;
	cellRenderSize.x /= (float) div.x;
	cellRenderSize.z /= (float) div.z;
}

void Projection::copyMaxTo(int* a) {
	for(int i = 0; i < div.x * div.z; i++) {
		a[i] = grid[i].max;
	}
}

void Projection::copyMaxTo(float* a) {
	for(int i = 0; i < div.x * div.z; i++) {
		a[i] = grid[i].max;
	}
}


void Projection::draw() {
	glBegin(GL_QUADS);
	float left = 0;
	float right = cellRenderSize.x;
	for(int x = 0; x < div.x; x++) {
		float back = 0;
		float front = cellRenderSize.z;
		for(int z = 0; z < div.z; z++) {
			ProjCell *cell = &grid[x + (z * div.x)];
			float totalP = (float) cell->max / (float) div.z;

				glColor3f(totalP,totalP,totalP);
				glVertex3f(left, maxLoc.y, back );
				glVertex3f(right,  maxLoc.y, back);
				glVertex3f(right, maxLoc.y, front);
				glVertex3f(left, maxLoc.y, front);
			
				back = front;
				front += cellRenderSize.z;
		}
		left = right;
		right+=cellRenderSize.x;
	}
	glEnd();

}
void Projection::convole(float *src, float *dst, int w, int h, float *conv, int cWidth, int cHeight, bool mirrorBorder) {

	size_t srcMemSize = sizeof(float) * w * h;
	float *d_src = NULL;
	cutilSafeCall( cudaMalloc((void**)&d_src, srcMemSize) ); // need to look up safecall TODO
	cutilSafeCall( cudaMemcpy(d_src, src, srcMemSize, cudaMemcpyHostToDevice) );


	float *d_dst = NULL;
	cutilSafeCall( cudaMalloc((void**)&d_dst, srcMemSize) ); // need to look up safecall TODO
	cudaMemset(d_dst, 0, sizeof(float) * srcMemSize);


	float *d_conv = NULL;
	size_t convMemSize = sizeof(float) *cWidth * cHeight;
	cutilSafeCall( cudaMalloc((void**)&d_conv, convMemSize) ); // need to look up safecall TODO
	cutilSafeCall( cudaMemcpy(d_conv, conv, convMemSize, cudaMemcpyHostToDevice) );


	
	gpu_conv(d_src, d_dst, w, h, d_conv, cWidth, cHeight, mirrorBorder);

	
	cutilSafeCall( cudaMemcpy(dst, d_dst, srcMemSize, cudaMemcpyDeviceToHost) );

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_conv);

}
