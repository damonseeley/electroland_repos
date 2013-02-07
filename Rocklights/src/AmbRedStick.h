

#ifndef __AMBREDSTICK_H__
#define __AMBREDSTICK_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"



class AmbRedStick  : public Ambient {
public:
  static int red1[];
  static int red2[];
  static int red3[];
  static int red4[];
  static int red1T[];
  static int red2T[];
  static int red3T[];
  static int red4T[];

   float densityFactor;
   float targetDensityFactor;
   float speedFactor;

  AmbRedStick(bool ci) ;
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);



} 
;

class AmbCRedStick : public AmbientCreator {
public:
	AmbCRedStick(void) {};
	~AmbCRedStick(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbRedStick(ci);
	}
	virtual string getName() { return "AmbRedStick"; };

};
#endif