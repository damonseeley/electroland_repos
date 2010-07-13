#include "VoxelBuilder.h"
#include <cuda_runtime.h>
#include <cutil_inline.h>

extern "C" void gpu_calcVoxel(int pointCnt, float* pixels, float minX, float minY, float minZ, float maxX, float maxY, float maxZ, int divX, int divY, int divZ, int* voxGrid);




VoxelBuilder::VoxelBuilder(Vec3f minDim, Vec3f maxDim, Vec3i maxDivisions, int levels) {
		this->minDim = minDim;
		this->maxDim = maxDim;
		this->maxDivisions = maxDivisions;
		this->levels = levels;
		voxels = (Voxel **) malloc(sizeof(Voxel *) * levels);
		Vec3i curDivisions = maxDivisions;
		for(int i = 0; i < levels; i++) {
			voxels[i] = new Voxel(minDim, maxDim, curDivisions);
			float div = curDivisions.x / .5f;
			curDivisions.x = (int) ceil(div); 
			div = curDivisions.y / .5f;
			curDivisions.y = (int) ceil(div); 
			div = curDivisions.z / .5f;
			curDivisions.z = (int) ceil(div); 
		}
}


void VoxelBuilder::calcVoxels(int pointCnt, float* points) {
	size_t pointsize = pointCnt * 3 * sizeof(float);
	float* d_points;
	cutilSafeCall( cudaMalloc((void**)&d_points, pointsize));
	cutilSafeCall( cudaMemcpy(d_points, points, pointsize, cudaMemcpyHostToDevice) );
	
	for(int i = 0; i < this->levels; i++) {
		Voxel *vox = voxels[i];
		vox->clear();
		size_t gridSize = vox->gridSize * sizeof(int);
		int* d_voxGrid;
		cutilSafeCall( cudaMalloc((void**)&d_voxGrid, sizeof(int) * gridSize));
		cutilSafeCall( cudaMemcpy(d_voxGrid, vox->getGrid(), gridSize, cudaMemcpyHostToDevice) );


	}


}