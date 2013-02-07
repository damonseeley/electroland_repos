#include "Agmt1SquareGreenKO.h"

int Agmt1SquareGreenKO::green[] = { -10000, 255,-10000, -10000,255,-10000, 1000, -1};


void Agmt1SquareGreenKO::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::GREEN);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), green, -1);


}
