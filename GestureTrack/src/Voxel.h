#ifndef __VOXEL_H__
#define __VOXEL_H__

#include "CinderVector.h"
#include "CloudConstructor.h"

#include <GL/glut.h>

using namespace cinder;

class Voxel {
public:
	static GLuint displayList;
	Vec3f minDim;
	Vec3f maxDim;
	Vec3i divisions;
	int gridSize;
	float *grid;
	float *d_vox;

//	static float* glFloorPoints;
//	static float* glFloorColors;
//	static float floorGridPointCnt;

	Voxel(Vec3f minDim, Vec3f maxDim, Vec3i divisions, bool createDisplayList = true);

	float* getGrid() { return grid; };
	void clear() { memset(grid, 0, gridSize * sizeof(int));};
	~Voxel();
    
	void calcVoxel(int pointCloudCnt,  float* pointCloud,  bool cloudIsOnGPU = false, bool freeSelfFromGPU = true); 

	virtual void createDisplayList();

	virtual void draw(float renderThresh=1.0f);
//	void constructFloorPoints();

	void copyGrid(Voxel *vox); // assumes same dims, only copies grid

	void addScale(float a, Voxel *vox, float b, bool freeFromGPU = true); // result is a * this[i] + b * vox[i]
	void mult(Voxel *vox, bool freeFromGPU = true); // result is  this[i] * vox[i]
	void add(Voxel *vox, bool freeFromGPU = true); // result is  this[i] + vox[i]
	void sub(Voxel *vox, bool freeFromGPU = true); // result is  this[i] * vox[i]
	void sub(Voxel *a, Voxel *b, bool freeFromGPU = true); // result is  a-b
	void setMask(Voxel *src, Voxel *mask, float thresh, bool freeFromGPU= true);
	void setNoiseFilter(Voxel *src, float thresh, bool freeFromGPU= true);
	void thresh(float t, bool freeFromGPU = true); // result is a * this[i] + b * vox[i]
	void scalarMult(float v, bool freeFromGPU = true); 
	void threshSet(float t, float below, float above, bool freeFromGPU = true);
	void incIfOverThresh(Voxel *other, float t, bool freeFromGPU=true); // if other[i] > t then this[i]++ else 0
	void scaleDownFrom(Voxel *doubleSize,  bool freeFromGPU=true);



	void allocateGridOnGPU(bool copy = true); // if copy is false valus are zeroed
	void deallocateGridOnGPU();
	float* getGridOnGPU();
	size_t voxMemSize;

	void scaleDownFrom_kernel(int gridSize, int dx, int dy, int dz, float *d_this, float *d_that);

};

#endif
