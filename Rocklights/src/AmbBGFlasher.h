#ifndef __AMBBGFLASHER_H__
#define __AMBBGFLASHER_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"




class AmbBGFlasher : public Ambient {
public:

	float flashProb;
	int flashAttempPerFrame;

	  AmbBGFlasher(bool ci) ;
	  void flash(int p, int c, int r);
	 
	  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) ;


} 
;
class AmbCBGFlasher : public AmbientCreator {
public:
	AmbCBGFlasher(void) {};
	~AmbCBGFlasher(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbBGFlasher(ci);
	}
	virtual string getName() { return "BGFlasher"; };

};
#endif