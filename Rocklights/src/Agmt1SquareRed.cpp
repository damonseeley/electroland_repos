#include "Agmt1SquareRed.h"

int Agmt1SquareRed::red[] = { 255, 0,0, 255,0,0, 1000, -1};


void Agmt1SquareRed::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::RED);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), red, -1);

}
