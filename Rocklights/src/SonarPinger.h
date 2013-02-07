#pragma once
#include "avatar.h"

class SonarPinger :
	public Avatar
{
public:
	SonarPinger(void);
	~SonarPinger(void);
};

#ifndef __AV1SQUARE_H
#define __AV1SQUARE_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "SoundHash.h"
#include "AvatarCreator.h"

class Avatar; 

class AV1Square : public Avatar  {
static int red[]; 
static int green[]; 
static int blue[]; 

static int curColor;

enum { RED, GREEN, BLUE };

public:
  AV1Square(PersonStats *personStats,  Interpolators *interps);
  ~AV1Square();
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps); 
}
;

class AC1Square : public AvatarCreator {
public:
	AC1Square(void) {};
	~AC1Square(void) {};
	
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) {
		return new AV1Square(personStats, interps);
	}

	virtual string getName() { return "AV1Square"; };

};
#endif