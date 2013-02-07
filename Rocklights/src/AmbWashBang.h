#ifndef __AMBWASHBANG_H__
#define __AMBWASHBANG_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"



class AmbWashBang : public Ambient {
public:
	static int wash[];
	float washSpeedScale;
	int timeToStab;




  AmbWashBang(bool ci) ;
  ~AmbWashBang() ;
  void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);

} 
;

class AmbCWashBang : public AmbientCreator {
public:
	AmbCWashBang() {};
	~AmbCWashBang() {};
	
	virtual Ambient* create(bool ci) {
		return new AmbWashBang(ci);
	}
	virtual string getName() { return "WashBang"; };

};
#endif