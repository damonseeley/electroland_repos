#ifndef __AMBTARGETFLASH_H__
#define __AMBTARGETFLASH_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"

class AmbTargetFlash  : public Ambient {
  static int flashAndFade[]; 

public:
  int targetFlashHoldTime ;
  int curTarget;
  int holdTimeLeft;
  int oldTarget;
  
  AmbTargetFlash(bool ci) ;
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);


} 
;
class AmbCTargetFlash : public AmbientCreator {
public:
	AmbCTargetFlash(void) {};
	~AmbCTargetFlash(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbTargetFlash(ci);
	}
	virtual string getName() { return "TargetFlash"; };

};
#endif