#include "AgmtBigCross.h"

int AgmtBigCross::red[] = { 255, 0,0, 0,0,0, 1000, -1};
int AgmtBigCross::green[] = { 0, 255,0, 0,0,0, 1000, -1};
int AgmtBigCross::blue[] = { 0, 0,255, 0,0,0, 1000, -1};


void AgmtBigCross::apply(Avatar *avatar, Interpolators *interps) {
		int* colorIntp;
	switch(avatar->getColor()) {
	case Avatar::RED:
		colorIntp = red;
		break;
	case Avatar::GREEN:
		colorIntp = green;
		break;
	case Avatar::BLUE:
		colorIntp = blue;
		break;
	}
		int w = Panels::thePanels->getWidth(Panels::A);
		int h = Panels::thePanels->getHeight(Panels::A);
		for(int i = -w; i < w; i++) {
			new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, i, 0), colorIntp,1);
		}
		for(int i = -h; i < h; i++) {
			new IGeneric(interps, avatar->addOffsetPixel(Avatar::A, 0, i), colorIntp, 1);
		}
	

}
