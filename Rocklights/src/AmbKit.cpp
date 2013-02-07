#include "AmbKit.h"

int AmbKit::kit1[] = { 
	255, 0, 0, 255, 0, 0, 1000,
	85, 0, 0, 85, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
    -1
};

int AmbKit::kit2[] = { 
	85, 0, 0, 85, 0, 0, 1000,
	255, 0, 0, 255, 0, 0, 1000,
	85, 0, 0, 85, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
	25, 0, 0, 25, 0, 0, 1000,
	255, 0, 0, 255, 0, 0, 1000,
    -1
};
int AmbKit::blueKit1[] = { 
	0,0,255,0,  0, 255,  1000,
	0,0,85, 0, 0, 85,  1000,
	0,0,25, 0, 0, 25,  1000,
	0,0,25, 0, 0, 25,  1000,
	0,0,25, 0, 0, 25,  1000,
	0,0,25, 0, 0, 25,  1000,
    -1
};

int AmbKit::blueKit2[] = { 
	 0, 0, 85, 0, 0, 85,1000,
	 0, 0, 255, 0, 0, 255,  1000,
	 0, 0, 85, 0, 0, 85,  1000,
	 0, 0, 25, 0, 0, 25,  1000,
	 0, 0, 25, 0, 0, 25,  1000,
	 0, 0, 255, 0, 0, 255,   1000,
    -1
};

void AmbKit::setup(bool isRed, bool lockStep) {
  for(int p = 0; p < Panels::PANEL_CNT; p++) {    
    for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
      for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
		  if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
			  float speedVar = 10.0f;
			  if(! lockStep) {
				speedVar = (((float)random(200)) / 100.0) + 8.0f;
			  }
			  if(isRed) {
			  (new IGeneric(interps, addAmbientPixel(p, c, r, 0), kit1, -1, 0))->timeScale = speedVar;
			(new IGeneric(interps, addAmbientPixel(p, c, r, 1), kit2, -1, 0))->timeScale = speedVar;
			(	new IGeneric(interps, addAmbientPixel(p, c, r, 2), kit2, -1, 3))->timeScale =speedVar;
			(	new IGeneric(interps, addAmbientPixel(p, c, r, 3), kit1, -1, 3))->timeScale = speedVar;
			  } else {
			  (new IGeneric(interps, addAmbientPixel(p, c, r, 0), blueKit1, -1, 0))->timeScale = speedVar;
			(new IGeneric(interps, addAmbientPixel(p, c, r, 1), blueKit2, -1, 0))->timeScale = speedVar;
			(	new IGeneric(interps, addAmbientPixel(p, c, r, 2), blueKit2, -1, 3))->timeScale =speedVar;
			(	new IGeneric(interps, addAmbientPixel(p, c, r, 3), blueKit1, -1, 3))->timeScale = speedVar;
			  }
			}
		  }
	  }
	}
  }
