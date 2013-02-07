#ifndef __AGMTBLUELINEKO_H__
#define __AGMTBLUELINEKO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtBlueLineKO : public Arrangement {
public:

	AgmtBlueLineKO(){};
	~AgmtBlueLineKO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "BlueLineKOH"; };
	void setup(Avatar *avatar, Interpolators *interps, bool horz);
}
;
class AgmtBlueLineKOV : public AgmtBlueLineKO {
public:

	AgmtBlueLineKOV(){};
	~AgmtBlueLineKOV(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "BlueLineKOV"; };
}
;
#endif