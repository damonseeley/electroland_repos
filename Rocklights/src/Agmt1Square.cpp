#include "Agmt1Square.h"

int Agmt1Square::red[] = {		255, 0,0,	255,0,0, 1000, -1};
int Agmt1Square::blue[] = {		0, 0,255,	0,0,255, 1000, -1};
int Agmt1Square::green[] = {	0, 255,0,	0,255,0, 1000, -1};


void Agmt1Square::apply(Avatar *avatar, Interpolators *interps) {

	switch(avatar->getColor()) {
	case Avatar::RED:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), red, -1);
	break;
	case Avatar::BLUE:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), blue, -1);
	break;
	case Avatar::GREEN:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), green, -1);
	break;
	}


}
