#ifndef __AVATARCREATER_H__
#define __AVATARCREATER_H__

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include <string>

class AvatarCreator
{
public:
		  enum {AVATAR, GENERIC};
		  int type;

public:
	AvatarCreator() { type = AVATAR; }
	~AvatarCreator(){}
	virtual Avatar* create(PersonStats *personStats,  Interpolators *interps) = 0;
	virtual string getName() = 0;
	
};

#endif
