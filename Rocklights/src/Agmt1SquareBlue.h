#ifndef __AGMT1SQUAREBLUE_H__
#define __AGMT1SQUAREBLUE_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareBlue : public Arrangement {
public:
	static int blue[]; 

	Agmt1SquareBlue(){};
	~Agmt1SquareBlue(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "BlueSquare"; };
}
;

#endif