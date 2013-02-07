#include "AV4Square.h"

int AV4Square::red[] = {
    255,  -255,    -255,    255, -255, -255, 1000, 
  -1};


 int AV4Square::redAnim[] = {
    0,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    0, 0, 0, 200, 
    255,  -255,    -255,    255, -255, -255, 200, 
  -1};

 int AV4Square::redAnim3Dots[] = {
    0,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    0, 0, 0, 200,  
    255,  -255,    -255,    255, -255, -255, 200, 
  -1};

AV4Square::AV4Square(PersonStats *personStats, Interpolators *interps) : Avatar () {

  new IGeneric(interps, addOffsetPixel(A, 0, 0), redAnim, -1, 3);
  new IGeneric(interps, addOffsetPixel(A, -1, 0), redAnim, -1, 2);
  new IGeneric(interps, addOffsetPixel(A, 0, -1), redAnim, -1, 0);
  new IGeneric(interps, addOffsetPixel(A, -1, -1), redAnim, -1, 1);
 
}


void AV4Square::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {

  int pil = personStats->nearPillar;
  int row = 0;

  if (pil != 0) {
    if (pil < 0) {
      row = -pil;
      pil = 1;
    }
  

    for (int col = 0; col < 7; col++) {
      Panels::thePanels->getPixel(pil, col, row)->addColor(255, 0, 0);
    }

  }
}