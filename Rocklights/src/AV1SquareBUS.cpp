#include "AV1SquareBUS.h"



int AV1SquareBUS::red[] = {
  255,0,0 ,  255,0,0, 1000
    -1
}
;
  int AV1SquareBUS::blueFade[] = {
    0,  0,    0,    0, 0, 255, 300,
	0,  0,    255,    0, 0, 0, 300, 
-1
};



    AV1SquareBUS::AV1SquareBUS(PersonStats *personStats, Interpolators *interps) : Avatar () {

new IGeneric(interps, addOffsetPixel(A, 0, 0, 0), red, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 1), red, -1, 0.5f);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 2), red, -1, 1.0f);
  new IGeneric(interps, addOffsetPixel(A, 0, 0, 3), red, -1, 1.5f);

  new IGeneric(interps, addOffsetPixel(A, 1, -1,0), blueFade, -1, 0.0f);
    new IGeneric(interps, addOffsetPixel(A, 1, -1,1), blueFade, -1, 0.5f);
	new IGeneric(interps, addOffsetPixel(A, 1, -1,2), blueFade, -1, 1.0f);
	new IGeneric(interps, addOffsetPixel(A, 1, -1,3), blueFade, -1, 1.5f);

  new IGeneric(interps, addOffsetPixel(A, 1, 0,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, 1, 0,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, 1, 0,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, 1, 0,3), blueFade, -1, 1.5f);

  new IGeneric(interps, addOffsetPixel(A, 1, 1,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, 1, 1,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, 1, 1,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, 1, 1,3), blueFade, -1, 1.5f);

  new IGeneric(interps, addOffsetPixel(A, 0, 1,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, 0, 1,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, 0, 1,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, 0, 1,3), blueFade, -1, 1.5f);
  
  new IGeneric(interps, addOffsetPixel(A, 0, -1,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, 0, -1,1), blueFade, -1, 0.5f);
  new IGeneric(interps, addOffsetPixel(A, 0, -1,2), blueFade, -1, 1.0f);
  new IGeneric(interps, addOffsetPixel(A, 0, -1,3), blueFade, -1, 1.5f);
  
  new IGeneric(interps, addOffsetPixel(A, -1, -1,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, -1, -1,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, -1, -1,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, -1, -1,3), blueFade, -1, 1.5f);
  
  new IGeneric(interps, addOffsetPixel(A, -1, 0,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, -1, 0,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, -1, 0,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, -1, 0,3), blueFade, -1, 1.5f);
  
  new IGeneric(interps, addOffsetPixel(A, -1, 1,0), blueFade, -1, 0.0f);
  new IGeneric(interps, addOffsetPixel(A, -1, 1,1), blueFade, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, -1, 1,2), blueFade, -1, 1.0f);
	  new IGeneric(interps, addOffsetPixel(A, -1, 1,3), blueFade, -1, 1.5f);


 
}
