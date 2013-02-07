#ifndef __AMBCOLORWASH_H__
#define __AMBCOLORWASH_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"

class AmbColorWash : public Ambient {
public:
	int* colorIntp;
	AmbColorWash(bool ci);
	~AmbColorWash(void) { delete[] colorIntp; }
	virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);
};


class AmbCColorWash : public AmbientCreator {
public:
	AmbCColorWash(void) {};
	~AmbCColorWash(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbColorWash(ci);
	}
	virtual string getName() { return "ColorWash"; };

};
#endif