#ifndef __AGMTSQUAREPULSE_H__
#define __AGMTSQUAREPULSE_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtSquarePulse : public Arrangement {
public:
	 static int redPulse[];


	AgmtSquarePulse(){};
	~AgmtSquarePulse(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "SquarePulse"; };
}
;
#endif