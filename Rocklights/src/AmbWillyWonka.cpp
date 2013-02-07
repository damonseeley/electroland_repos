#include "AmbWillyWonka.h"


int AmbWillyWonka::red[] = { 
	100,0,	0,	100,	0, 0, 1000,
	100,	0,	0,	255,0, 0, 1000,
	255, 0, 0,	255,0, 0, 250,
	255, 0, 0,	100,	0, 0, 1000,
    -1
};
int AmbWillyWonka::green[] = { 
	0,	100,	0,		0,	100,	0,		1000,
	0,	100,	0,		0,	255,0,		1000,
	0,	255,0,		0,	255,0,		250,
	0,	255,0,		0,	100,	0,		1000,
    -1
};
int AmbWillyWonka::blue[] = { 
	0,	0,	100,		0,	0,	100,		1000,
	0,	0,	100,		0,	0,	255,	1000,
	0,	0,	255,	0,	0,	255,	250,
	0,	0,	255,	0,	0,	100,		1000,
    -1
};



AmbWillyWonka::AmbWillyWonka(bool ci) : Ambient(ci) {
  for(int p = 0; p < Panels::PANEL_CNT; p++) {
  for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
    for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
      for(int s = 0; s < 4; s++) { 
        if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
			float start = ((float) random(10)) / 10.0f;
			float timeScale = (((float) random(100))/ 100.0f);
			(new IGeneric(interps, addAmbientPixel(p, c, r, s), red, -1, start))->timeScale = timeScale;
			 start = ((float) random(10)) / 10.0f;
			 timeScale = (((float) random(100))/ 100.0f) ;
			(new IGeneric(interps, addAmbientPixel(p, c, r, s), green, -1, start))->timeScale = timeScale;
			 start = ((float) random(10)) / 10.0f;
			 timeScale = (((float) random(100))/ 100.0f) ;
			(new IGeneric(interps, addAmbientPixel(p, c, r, s), blue, -1, start))->timeScale = timeScale;
        }
      }
    }
  }
  }
}