#ifndef __AVSINGLEPIXELDANCE_H__
#define __AVSINGLEPIXELDANCE_H__

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "AvatarCreator.h"

class Avatar; 

class AVSinglePixelDance : public Avatar  {
  static int red[];
  static int redFade[];
  int oldRow;
  int oldCol;
  int oldPil;



  static int redIn6[];
  static int redIn5[];
  static int redIn4[];
  static int redIn3[];
  static int redIn2[];
  static int redIn1[];
  static int redIn0[];

  IGeneric *c0;
  IGeneric *c1;
  IGeneric *c2;
  IGeneric *c3;
  IGeneric *c4;
  IGeneric *c5;
  IGeneric *c6;


public:
  AVSinglePixelDance(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);
}
;
class ACSinglePixelDance : public AvatarCreator {
public:
	ACSinglePixelDance(void) {  };
	~ACSinglePixelDance(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AVSinglePixelDance(personStats, interps);
	}

	virtual string getName() { return "SinglePixelDance"; };

};
#endif