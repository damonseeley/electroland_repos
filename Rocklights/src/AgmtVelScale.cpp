#include "agmtvelscale.h"

int AgmtVelScale::red[] = {
	255,	0,	0,	255,	0,	0,	250,
	255,	0,	0,	0,		0,	0,	100,
	-1};

void AgmtVelScale::apply(Avatar *avatar, Interpolators *interps) {
	int col = avatar->getCol();
	int row = avatar->getRow();

	for(int c =  -1; c <= 1; c++) {
		for(int r =  -1; r <=  1; r++) {
			new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, c, r), red);
		}
	} 
}

