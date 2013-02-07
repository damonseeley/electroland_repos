#ifndef __AGMTVELSCALE_H__
#define __AGMTVELSCALE_H__

#include "Arrangement.h"
#include "IGeneric.h"

class AgmtVelScale : public Arrangement {
public:
	static int red[];

	AgmtVelScale(){};
	~AgmtVelScale(){};
	virtual void apply(Avatar *avater, Interpolators *interps);

	virtual string getName() { return "VelScale"; };
}
;

#endif