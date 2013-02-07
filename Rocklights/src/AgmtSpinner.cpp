#include "AgmtSpinner.h"
#include "Agmt1Square.h"


int AgmtSpinner::spinnerBlue[]  = 
{ 0, 0,0, 0,0,0, 100, //0
0, 0,0, 0,0,0, 100,   //1
0, 0,0, 0,0,0, 100,  //2
0, 0,0, 0,0,0, 100, //3
0, 0,0, 0,0,0, 100, //4
0, 0,0, 0,0,0, 100, //5
0, 0,0, 0,0,0, 100, //6
	0,0,255, 0,0,255, 100, //7
	-1};

void AgmtSpinner::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::RED);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 0), Agmt1Square::red, -1);

	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, -1), spinnerBlue, -1, 0);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 0 ), spinnerBlue, -1, 1);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, -1, 1), spinnerBlue, -1, 2);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, 1), spinnerBlue, -1, 3);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 1), spinnerBlue, -1, 4);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, 0), spinnerBlue, -1, 5);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 1, -1), spinnerBlue, -1, 6);
	new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, -1), spinnerBlue, -1, 7);
}
