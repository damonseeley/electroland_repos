#ifndef __FLOOR_H__
#define __FLOOR_H__

#include "CinderVector.h"

using namespace cinder;
class Floor {

public:
	float level;
	float minX;
	float maxX;
	float depth;
	Vec3f backColor;
	Vec3f frontColor;

	Floor(float level, float minX, float maxX, float depth, Vec3f backColor, Vec3f frontColor) ;
	void render();

}
;
#endif