#ifndef __VOXEL_BUILDER_H__
#define __VOXEL_BUILDER_H__

#include "Voxel.h"

using namespace elvector;

class VoxelBuilder {
public:
	Vec3f minDim;
	Vec3f maxDim;
	Vec3i maxDivisions;
	int levels;
	Voxel **voxels;

	VoxelBuilder(Vec3f minDim, Vec3f maxDim, Vec3i maxDivisions, int levels);
	void calcVoxels(int pointCnt, float* points);
	Voxel* getVoxels(int level) { return voxels[level];};

	
};

#endif
