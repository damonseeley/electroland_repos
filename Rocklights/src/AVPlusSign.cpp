#include "AVPlusSign.h"

int AVPlusSign::red[] = {
    255,  0,    0,    255, 0, 0, 1000, 
  -1};
int AVPlusSign::green[] = {
    0,  255,    0,    0, 255, 0, 1000, 
  -1};
int AVPlusSign::blue[] = {
    0,  0,    255,    0,  0, 255, 1000, 
  -1};

AVPlusSign::AVPlusSign(PersonStats *personStats, Interpolators *interps) : Avatar () {
setColor(personStats->color);

  switch(personStats->color) {
  case RED:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), red, -1);
  new IGeneric(interps, addOffsetPixel(A, 1, 0), red, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, 1), red, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, 0), red, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, -1), red, -1);
    break;
  case GREEN:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), green, -1);
  new IGeneric(interps, addOffsetPixel(A, 1, 0), green, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, 1), green, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, 0), green, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, -1), green, -1);
    break;
  case BLUE:
  new IGeneric(interps, addOffsetPixel(A, 0, 0), blue, -1);
  new IGeneric(interps, addOffsetPixel(A, 1, 0), blue, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, 1), blue, -1);
  new IGeneric(interps, addOffsetPixel(A, -1, 0), blue, -1);
  new IGeneric(interps, addOffsetPixel(A, 0, -1), blue, -1);
    break;
  }
 
}


void AVPlusSign::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
/*
   int pil = personStats->nearPillar;
  int row = 0;

  if (pil != 0) {
    if (pil < 0) {
      row = -pil;
      row -= 1;
      pil = 1;
    }
  

    for (int col = 0; col < 7; col++) {
      Panels::thePanels->getPixel(pil, col, row)->addColor(255, 0, 0);
    }

  }
  */
}