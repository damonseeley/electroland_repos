#ifndef __AMBBLINKTAG_H__
#define __AMBBLINKTAG_H__

#include "Ambient.h"
#include "AmbientCreator.h"
#include "StringSeq.h"
#include "PersonStats.h"
class AmbBlinkTag  : public Ambient {
public:
	enum { ON, OFF, DELAY };
	int blinkOnTime;
	int blinkOffTime;
	int blinkTimes;
	int intraBlinkDelay;
	int timeLeft;
	int blinksLeft;
	int state;
	int style;
	unsigned long curId;
	StringSeq *sounds;
	



  AmbBlinkTag(bool ci) ;
  ~AmbBlinkTag() ;
  void reset();
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);


} 
;

class AmbCBlinkTag : public AmbientCreator {
public:
	AmbCBlinkTag(void) {};
	~AmbCBlinkTag(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbBlinkTag(ci);
	}
	virtual string getName() { return "BlinkTag"; };

};
#endif