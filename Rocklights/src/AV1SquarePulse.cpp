#include "AV1SquarePulse.h"

int AV1SquarePulse::redPulse[] = {
    150,  0,    100,    255, 0, 0, 500, 
    255,  0,    0,    150, 0, 100, 500, 
  -1};

AV1SquarePulse::AV1SquarePulse(PersonStats *personStats, Interpolators *interps) : Avatar () {
  new IGeneric(interps, addOffsetPixel(A, 0, 0), AV1SquarePulse::redPulse, -1);
 
}
