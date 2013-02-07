#ifndef __AGMTPLUS_H__
#define __AGMTPLUS_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtPlus : public Arrangement {
public:

//	static int red[]; 

	AgmtPlus(){};
	~AgmtPlus(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "Plus"; };
}
;

#endif