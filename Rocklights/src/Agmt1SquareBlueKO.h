#ifndef __AGMT1SQUAREBLUEKO_H__
#define __AGMT1SQUAREBLUEKO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareBlueKO : public Arrangement {
public:
	static int blue[]; 

	Agmt1SquareBlueKO(){};
	~Agmt1SquareBlueKO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "BlueSquareKnockOut"; };
}
;

#endif