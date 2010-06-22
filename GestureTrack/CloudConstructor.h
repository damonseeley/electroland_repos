#ifndef __CLOUD_CONSTRUCTOR_H__
#define __CLOUD_CONSTRUCTOR_H__

#include "TyzxCam.h"


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

	CloudConstructor(TyzxCam *cams[], int camCount);
	void calcPoints();
	~CloudConstructor();
};

#endif
