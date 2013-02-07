#include "AgmtX.h"
#include "Agmt1Square.h"

//int Agmt1Square::red[] = { 255, 0,0, 255,0,0, 1000, -1};


void AgmtX::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::RED);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 1), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 1), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, -1), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, -1), Agmt1Square::red, -1);
}
