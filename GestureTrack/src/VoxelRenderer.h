#ifndef __VOXEL_RENDERER_H__
#define __VOXEL_RENDERER_H__


#include "Voxel.h"
#include <windows.h>
#include <GL/glut.h>

#define FRONT	0
#define BACK	1
#define LEFT	2
#define RIGHT	3
#define TOP		4
#define BOT		5


class VoxelRenderer {
public:
	int to;
	int from;

	float cubePoints[4*6*3];
	float colorVals[4*6*3];

	Voxel *voxel;
	Vec3f colors[2][6];

	VoxelRenderer(Voxel *voxel);
	void constructDisplayList();
	void constructColorList(int slice);

	void setFrontColor(Vec3f f, Vec3f back, Vec3f l, Vec3f r, Vec3f t, Vec3f bot);
	void setBackColor (Vec3f f, Vec3f back, Vec3f l, Vec3f r, Vec3f t, Vec3f bot);

	void setVoxel(Voxel *voxel);
	void draw(DWORD curTime, float dt, float thresh);
	void setFromTo(float from, float to);
};
#endif