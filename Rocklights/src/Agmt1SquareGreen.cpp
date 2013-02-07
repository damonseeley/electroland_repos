#include "Agmt1SquareGreen.h"

int Agmt1SquareGreen::green[] = { 0, 255,0, 0,255,0, 1000, -1};


void Agmt1SquareGreen::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::GREEN);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), green, -1);


}
