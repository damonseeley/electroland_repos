#ifndef __AXIS_H__
#define __AXIS_H__

#include "CinderVector.h"

using namespace cinder;
class Axis {

public:
	Vec3f pos;
	Vec3f rot;

	Axis(Vec3f pos, Vec3f rot) { this->pos = pos; this->rot = rot;}
	void render();

}
;
#endif