#include "AV1Square.h"

int AV1Square::red[] = { 255, 0,0, 255,0,0, 1000, -1};
int AV1Square::green[] = { 0, 255,0, 0,255,0, 1000, -1};
int AV1Square::blue[] = { 0, 0,255, 0,0,255, 1000, -1};



AV1Square::AV1Square(PersonStats *personStats, Interpolators *interps) : Avatar () {

setColor(personStats->color);

  switch(personStats->color) {
  case RED:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), red, -1);
    break;
  case GREEN:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), green, -1);
    break;
  case BLUE:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), blue, -1);
    break;
  }


  // ENTER STUFF HERE
 
}


void AV1Square::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
}



AV1Square::~AV1Square() {
	// EXIT STUFF HERE
}