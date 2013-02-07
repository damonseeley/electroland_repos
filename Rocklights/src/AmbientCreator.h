#ifndef __AMBIENTCREATER_H__
#define __AMBIENTCREATER_H__

#include <string>
#include "Ambient.h"


class AmbientCreator
{
public:
	AmbientCreator() {}
	~AmbientCreator(){}
	virtual Ambient* create(bool createInterps) = 0;
	virtual string getName() = 0;
	
};

#endif
