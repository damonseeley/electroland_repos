#ifndef __AGMT1SQUAREREDKO_H__
#define __AGMT1SQUAREREDKO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareRedKO : public Arrangement {
public:
	static int red[]; 

	Agmt1SquareRedKO(){};
	~Agmt1SquareRedKO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "RedSquareKnockOut"; };
}
;

#endif