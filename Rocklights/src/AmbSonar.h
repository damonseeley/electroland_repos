#ifndef __AMBSonar_H__
#define __AMBSonar_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"

class AmbSonar : public Ambient {
	int msPerStick;
	int curStick;
	int timeLeftForStick;
	int width;
	int lastPing;
public:
		static int curCol;

public:
	AmbSonar(bool ci);
	~AmbSonar(void);
	 virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) ;

};



class AmbCSonar : public AmbientCreator {
public:
	AmbCSonar(void) {};
	~AmbCSonar(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbSonar(ci);
	}
	virtual string getName() { return "AmbSonar"; };

};

#endif;