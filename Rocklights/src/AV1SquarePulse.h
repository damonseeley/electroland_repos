#ifndef __AV1SQUAREPULSE_H
#define __AV1SQUAREPULSE_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "AvatarCreator.h"

class Avatar; 

class AV1SquarePulse : public Avatar  {

static int redPulse[];


public:
  AV1SquarePulse(PersonStats *personStats,  Interpolators *interps);
}
;
class AC1SquarePulse : public AvatarCreator {
public:
	AC1SquarePulse(void) {  };
	~AC1SquarePulse(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AV1SquarePulse(personStats, interps);
	}

	virtual string getName() { return "SquarePulse"; };

};
#endif