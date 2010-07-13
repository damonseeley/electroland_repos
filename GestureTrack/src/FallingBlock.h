#ifndef __FALLING_BLOCL_H__
#define __FALLING_BLOCL_H__

#include "CinderVector.h"


class FallingBlock : public Voxel {
public:
	static GLuint displayList;

	Vec3f location;
	Vec3f voxelIndex;

	FallingBlock(Vec3f location, Vec3f voxelIndex);

	void draw();


}

#endif