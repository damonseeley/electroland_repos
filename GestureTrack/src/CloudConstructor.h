#ifndef __CLOUD_CONSTRUCTOR_H__
#define __CLOUD_CONSTRUCTOR_H__

#include "TyzxCam.h"
#include "GestureTypeDefs.h"


class CloudConstructor {
public:

	bool isInitNeeded;

	int camCnt;
	TyzxCam**  cams;
	
	int  imgWidth;
	int  imgHeight;
	int imgSize;

	double	*camParams;

	int gridSize;

	int pointCnt;
	float* points;
	float* d_resultPoints;

	CloudConstructor(TyzxCam *cams[], int camCount);
	void calcPoints(bool freeResultPointsOnGPU = true);
	void getBounds(Vec3f  &min, Vec3f  &max);

	int getPointCnt() { return pointCnt; };
	float* getPoints() { return points; };

	void freeGPUPoints();
	float* getGPUPoints() {  return d_resultPoints; }; 

	~CloudConstructor();

	void cull(float ax, float az, float bx, float bz, float floor); 

};

#endif
