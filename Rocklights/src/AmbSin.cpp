#include "AmbSin.h"

int AmbSin::wave[] = { 
	0,0,255,	0,0,255, 1000,
	0,0,255,	0,0,200, 1000,
	0,0,200,	0,0,155, 1000,
	0,0,155,		0,0,55, 1000,
	0,0,55,		0,0,55, 1000,
	0,0,55,		0,0,155, 1000,
	0,0,155,		0,0,200, 1000,
	0,0,200,	0,0,255, 1000,
    -1
};


AmbSin::AmbSin(bool ci) : Ambient(ci) {
  for(int p = 0; p < Panels::PANEL_CNT; p++) {    
    for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c+=2) {
      for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
          if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
				new IGeneric(interps, addAmbientPixel(p, c, r, 0), wave, -1, 0);
				new IGeneric(interps, addAmbientPixel(p, c, r, 1), wave, -1, 1);
				new IGeneric(interps, addAmbientPixel(p, c, r, 2), wave, -1, 2);
				new IGeneric(interps, addAmbientPixel(p, c, r, 3), wave, -1, 4);
		  }
          if(! Panels::thePanels->panels[p].getPixel(c+1,r)->isTarget) {
				new IGeneric(interps, addAmbientPixel(p, c+1, r, 0), wave, -1, 4);
				new IGeneric(interps, addAmbientPixel(p, c+1, r, 1), wave, -1, 5);
				new IGeneric(interps, addAmbientPixel(p, c+1, r, 2), wave, -1, 6);
				new IGeneric(interps, addAmbientPixel(p, c+1, r, 3), wave, -1, 7);
		  }
	  }
	}
  }
}