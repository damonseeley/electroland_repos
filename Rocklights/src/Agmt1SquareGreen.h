#ifndef __AGMT1SQUAREGREEN_H__
#define __AGMT1SQUAREGREEN_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareGreen : public Arrangement {
public:
	static int green[]; 

	Agmt1SquareGreen(){};
	~Agmt1SquareGreen(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "GreenSquare"; };
}
;

#endif