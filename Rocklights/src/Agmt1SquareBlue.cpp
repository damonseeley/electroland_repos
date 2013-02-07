#include "Agmt1SquareBlue.h"

int Agmt1SquareBlue::blue[] = { 0, 0,255, 0,0,255, 1000, -1};


void Agmt1SquareBlue::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::BLUE);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), blue, -1);


}
