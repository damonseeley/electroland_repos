#include "AgmtO.h"
#include "Agmt1Square.h"

//int Agmt1Square::red[] = { 255, 0,0, 255,0,0, 1000, -1};


void AgmtO::apply(Avatar *a, Interpolators *interps) {
	a->setColor(Avatar::RED);
	
	new IGeneric(interps, a->addOffsetPixel(Panels::A, 1, 0), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, 0, 1), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, -1, 0), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, 0, -1), Agmt1Square::red, -1);
  
  new IGeneric(interps, a->addOffsetPixel(Panels::A, 1, 1), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, -1, -1), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, 1, -1), Agmt1Square::red, -1);
  new IGeneric(interps, a->addOffsetPixel(Panels::A, -1, 1), Agmt1Square::red, -1);
}
