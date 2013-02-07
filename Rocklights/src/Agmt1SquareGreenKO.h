#ifndef __AGMT1SQUAREGREENKO_H__
#define __AGMT1SQUAREGREENKO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareGreenKO : public Arrangement {
public:
	static int green[]; 

	Agmt1SquareGreenKO(){};
	~Agmt1SquareGreenKO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "GreenSquareKnockOut"; };
}
;

#endif