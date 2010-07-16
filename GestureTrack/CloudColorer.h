#ifndef __CLOUD_COLORER_H__
#define __CLOUD_COLORER_H__

#include "CinderVector.h"

using namespace cinder;


class CloudColorer {
	bool drawQuads;

public:
	float *colors;
	float *quads;
	float minZ;
	float diffZ;
	Vec3f minColor;
	Vec3f diffColor;
	bool hsv;
	float size;

	
	CloudColorer(float minZ, float maxZ, Vec3f minColor, Vec3f maxColor, bool hsv, float size);
	void calcColors(int pointCloudCnt,  float* pointCloud, bool cloudIsOnGPU = false);
	void calcQuads(int pointCloudCnt, float* pointCloud);
	float* getColors() { return colors; }
	~CloudColorer() { free(colors); free(quads); }
	void setQuads(bool b) { drawQuads = b; free(colors); free(quads); colors = NULL; quads = NULL; }

}
;
#endif