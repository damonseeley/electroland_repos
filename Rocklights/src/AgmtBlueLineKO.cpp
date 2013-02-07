#include "AgmtBlueLineKO.h"
#include "Agmt1SquareBlueKO.h"

void AgmtBlueLineKO::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::BLUE);
	setup(avatar, interps, true); 
}

void AgmtBlueLineKO::setup(Avatar *avatar, Interpolators *interps, bool horz) {
	if(horz) {
		int w = Panels::thePanels->getWidth(Panels::A);
		for(int i = -w; i < w; i++) {
			new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, i, 0), Agmt1SquareBlueKO::blue, -1);
		}
	} else {
		int h = Panels::thePanels->getHeight(Panels::A);
		for(int i = -h; i < h; i++) {
			new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, i), Agmt1SquareBlueKO::blue, -1);
		}
	}

}

void AgmtBlueLineKOV::apply(Avatar *avatar, Interpolators *interps) {
	avatar->setColor(Avatar::BLUE);
	setup(avatar, interps, false); 
}
