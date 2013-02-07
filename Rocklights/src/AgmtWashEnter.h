#ifndef __AGMTWASHENTER_H__
#define __AGMTWASHENTER_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtWashEnter : public Arrangement {
public:
	static int red[]; 
	static int green[]; 
	static int blue[]; 
	static int intensity;
	static int fadeTime;

	AgmtWashEnter();
	~AgmtWashEnter(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "WashEnter"; };
}
;

#endif