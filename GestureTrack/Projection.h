#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include "ProjCell.h"
#include "Voxel.h"
#include "CinderVector.h"


class Projection {
public:
	Vec3i div;

	ProjCell *grid;
	ProjCell *d_grid;

	size_t gridMemSize;

	Vec3f minLoc;
	Vec3f maxLoc;
	Vec3f cellRenderSize;



	Projection(Vec3i div);
	~Projection();

	void allocateGridOnGPU(bool copy = true); // if copy is false valus are zeroed
	void deallocateGridOnGPU();

	void calcProjection(Voxel *voxel, float thresh, bool freeGridFromGPU = true);

	void setRenderLocation(Vec3f min, Vec3f max);
	void draw();

	void copyMaxTo(int *a);
	void copyMaxTo(float *a);

	void convole(float *src, float *dst, int w, int h, float *conv, int cWidth, int cHeight, bool mirrorBorder) ;



}
;
#endif