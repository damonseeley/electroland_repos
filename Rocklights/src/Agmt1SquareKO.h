#ifndef __AGMT1SQUAREKO_H__
#define __AGMT1SQUAREKO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1SquareKO : public Arrangement {
public:
	static int negWhite[]; 

	Agmt1SquareKO(){};
	~Agmt1SquareKO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "TotalSquareKnockOut"; };
}
;

#endif