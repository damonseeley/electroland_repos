#ifndef __AV1SQUAREBUS_H
#define __AV1SQUAREBUS_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "AvatarCreator.h"

class Avatar; 

class AV1SquareBUS : public Avatar  {

static int blueFade[];
static int red[];


public:
  AV1SquareBUS(PersonStats *personStats,  Interpolators *interps);
}
;

class ACSquareBUS : public AvatarCreator {
public:
	ACSquareBUS(void) {  };
	~ACSquareBUS(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AV1SquareBUS(personStats, interps);
	}

	virtual string getName() { return "SquareBUS"; };

};
#endif
