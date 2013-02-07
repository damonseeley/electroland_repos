

#include "AV9Square.h"

int red2[] = {
    255,  0,    0,    255, 0, 0, 1000, 
  -1};

AV9Square::AV9Square(PersonStats *personStats, Interpolators *interps) : Avatar () {

  new IGeneric(interps, addOffsetPixel(A, 0, 0), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, 1, 0), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, 1), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, 0), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, -1), red2, -1);
  
  new IGeneric(interps, addOffsetPixel(A, 1, 1), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, -1), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, 1, -1), red2, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, 1), red2, -1);
 
}


void AV9Square::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {

  
}