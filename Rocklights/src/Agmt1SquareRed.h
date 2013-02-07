#ifndef __AGMT1SQUARERED_H__
#define __AGMT1SQUARERED_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareRed : public Arrangement {
public:
	static int red[]; 

	Agmt1SquareRed(){};
	~Agmt1SquareRed(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "RedSquare"; };
}
;

#endif