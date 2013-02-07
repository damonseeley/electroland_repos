#ifndef __AVGENERIC_H
#define __AVGENERIC_H

#include "Avatar.h"
#include "PersonStats.h"
#include "IGeneric.h"
#include "Interpolators.h"
#include "SoundHash.h"
#include "AvatarCreator.h"

class Avatar; 

class AVGeneric : public Avatar  {
	int loopNumber;

public:
  AVGeneric(PersonStats *personStats,  Interpolators *interps);
  ~AVGeneric();
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps); 
  void move(int col, int row, Interpolators *interps);
  void exit(Interpolators *interps);
  void enter(Interpolators *interps);
  void init(Interpolators *interps); // false means transition into state
  
}
;

class ACGeneric : public AvatarCreator {
public:
	ACGeneric(void) { 	type = GENERIC; };
	~ACGeneric(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AVGeneric(personStats, interps);
	}

	virtual string getName() { return "GENERIC"; };

};
#endif