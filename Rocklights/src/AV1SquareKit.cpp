#include "AV1SquareKit.h"

int AV1SquareKit::kit[] = {
  0,  0,    0,    0, 0, 0, 199, 
	0,  0,    0,    255, 0, 0, 199,
	255,  0,    0,    100, 0, 0, 199, 
	100,  0,    0,    0, 0, 0, 199, 
	0,  0,    0,    0, 0, 0, 199,
	0,  0,    0,    0, 0, 0, 199,
  -1
};


    AV1SquareKit::AV1SquareKit(PersonStats *personStats, Interpolators *interps) : Avatar () {


  new IGeneric(interps, addOffsetPixel(A, 0, 0, 0), kit, -1, 1);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 1), kit, -1, 2);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 2), kit, -1, 3);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 3), kit, -1, 4);
 
}
