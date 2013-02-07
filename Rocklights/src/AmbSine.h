

#ifndef __AMBSINE_H__
#define __AMBSINE_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"




class AmbSine : public Ambient {
public:
	int updateSpeed;
	int updateSpeed2;
	int updateTime;
	int updateTime2;
	int phase;
	int phase2;
	static int wave[];
  AmbSine(bool ci) ;
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);

} 
;
class AmbCSine : public AmbientCreator {
public:
	AmbCSine(void) {};
	~AmbCSine(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbSine(ci);
	}
	virtual string getName() { return "Sine"; };

};
#endif