#include "Agmt1SquareKO.h"

int Agmt1SquareKO::negWhite[] = { -10000, -10000,-10000, -10000,-10000,-10000, 1000, -1};


void Agmt1SquareKO::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::RED);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), negWhite, -1);


}
