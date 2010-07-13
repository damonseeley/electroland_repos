#ifndef PRESENCE_VOXEL_H
#define PRESENCE_VOXEL_H

#include "Voxel.h"

class PresenceVoxel : public Voxel {
public:
		PresenceVoxel(Vec3f minDim, Vec3f maxDim, Vec3i divisions, bool createDL = true);

		virtual void createDisplayList();
		virtual void draw(float renderThresh=1.0f);

};

#endif PRESENCE_VOXEL_H