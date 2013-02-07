#ifndef __ARRANGEMENT_H__
#define __ARRANGEMENT_H__

#include <string>
#include "Avatar.h"
#include "Interpolators.h"
#include "PersonStats.h"

class Avatar;
class Interpolators;
class PersonStats;

class Arrangement
{
public:
	Arrangement(){};
	~Arrangement(){};
	virtual void apply(Avatar *avater, Interpolators *interps) = 0;
	virtual void updateFrame(Avatar *avatar, PersonStats *personStats, int ct, int dt, Interpolators *interps) {} // only called on overhead
	virtual void exit(int c, int r, Interpolators *interps) {}

	virtual string getName() = 0;
}
;

#endif