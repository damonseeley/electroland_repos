#include "Agmt1SquareBlueKO.h"

int Agmt1SquareBlueKO::blue[] = { -10000, -10000,255, -10000,-10000,255, 1000, -1};


void Agmt1SquareBlueKO::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::BLUE);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), blue, -1);


}
