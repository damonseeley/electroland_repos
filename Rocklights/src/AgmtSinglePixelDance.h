#ifndef __AGMTSINGLEPIXELDANCE_H__
#define __AGMTSINGLEPIXELDANCE_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtSinglePixelDance : public Arrangement {
public:
	 static int red[];


	AgmtSinglePixelDance(){};
	~AgmtSinglePixelDance(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "SinglePixelDance"; };
}
;
#endif