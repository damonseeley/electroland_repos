#include "Agmt1SquareRedKO.h"

int Agmt1SquareRedKO::red[] = { 255, -10000,-10000, 255,-10000,-10000, 1000, -1};


void Agmt1SquareRedKO::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::RED);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), red, -1);
}
