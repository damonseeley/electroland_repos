#include "agmtsquarepulse.h"



int AgmtSquarePulse::redPulse[] = {
    150,  0,    100,    255, 0, 0, 500, 
    255,  0,    0,    150, 0, 100, 500, 
  -1};



void AgmtSquarePulse::apply(Avatar *avatar, Interpolators *interps) {
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), redPulse, -1);
}

