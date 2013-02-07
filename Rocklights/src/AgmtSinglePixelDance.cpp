#include "agmtsinglepixeldance.h"

int AgmtSinglePixelDance::red[] = {
  255,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    255, 0, 0, 200, 
    -1};



void AgmtSinglePixelDance::apply(Avatar *avatar, Interpolators *interps) {
		  new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0, 0), red, -1, 0.0f);
    new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0, 1), red, -1, .25f);
    new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0, 2), red, -1, 0.5f);
    new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0, 3), red, -1, .75f);
    //  new IGeneric(interps, addOffsetPixel(A, 0, -1), red, -1);
}

