#include "AgmtPlus.h"
#include "Agmt1Square.h"

//int Agmt1Square::red[] = { 255, 0,0, 255,0,0, 1000, -1};


void AgmtPlus::apply(Avatar *avatar, Interpolators *interps) {
//	avatar->setColor(avatar->getColor());
	switch(avatar->getColor()) {
	case Avatar::RED:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 0), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 1), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 0), Agmt1Square::red, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, -1), Agmt1Square::red, -1);
	break;
	case Avatar::BLUE:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), Agmt1Square::blue, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 0), Agmt1Square::blue, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 1), Agmt1Square::blue, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 0), Agmt1Square::blue, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, -1), Agmt1Square::blue, -1);
	break;
	case Avatar::GREEN:
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), Agmt1Square::green, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 0), Agmt1Square::green, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 1), Agmt1Square::green, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 0), Agmt1Square::green, -1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, -1), Agmt1Square::green, -1);
	break;
	}

}
