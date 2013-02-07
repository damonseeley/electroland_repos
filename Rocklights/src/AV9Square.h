
 
#ifndef __AV9SQUARE_H
#define __AV9SQUARE_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Panels.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "BasePixel.h"
#include "AvatarCreator.h"

class Avatar; 

class AV9Square : public Avatar  {
  
public:
  AV9Square(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);
};

class AC9Square : public AvatarCreator {
public:
	AC9Square(void) {};
	~AC9Square(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AV9Square(personStats, interps);
	}

	virtual string getName() { return "9Square"; };

};
#endif