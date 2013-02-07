#ifndef __AMBEMPTY_H__
#define __AMBEMPTY_H__

#include "Ambient.h"
#include "AmbientCreator.h"

class AmbEmpty :
	public Ambient
{
public:
	AmbEmpty(void);
	~AmbEmpty(void);
};

class AmbCAmbEmpty : public AmbientCreator { 
public:
	AmbCAmbEmpty(void) {};
	~AmbCAmbEmpty(void) {};
	
	virtual Ambient* create(bool cI) {
		return new Ambient(cI);
	}

	virtual string getName() { return "AmbEmpty"; };

};

#endif