#ifndef __AGMTSONARPINGER_H__
#define __AGMTSONARPINGER_H__

#include "Avatar.h"
#include "IGeneric.h"
#include "AvatarCreator.h"

class AVSonarPinger : public Avatar {
public:
	bool justPinged[MAXWAVES];

	static int red[];

	AVSonarPinger();
	~AVSonarPinger(){};
	 void apply(Avatar *avater, Interpolators *interps) {}
	void updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps);
}
;
class ACSonarPinger : public AvatarCreator {
public:
	ACSonarPinger(void) { 	 };
	~ACSonarPinger(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AVSonarPinger();
	}

	virtual string getName() { return "SonarPinger"; };

};

#endif