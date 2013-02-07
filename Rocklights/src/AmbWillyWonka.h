#ifndef __AMBWILLYWONKA_H__
#define __AMBWILLYWONKA_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"

class AmbWillyWonka  : public Ambient {
public:
  static int red[];
  static int green[];
  static int blue[];

  AmbWillyWonka(bool ci) ;
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {}


} 
;

class AmbCWillyWonka : public AmbientCreator {
public:
	AmbCWillyWonka(void) {};
	~AmbCWillyWonka(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbWillyWonka(ci);
	}
	virtual string getName() { return "WillyWonka"; };

};
#endif