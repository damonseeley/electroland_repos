#include "Ambient.h"
#include "MasterController.h"
Ambient::Ambient (bool swapAmbients) {   
		caller = NULL;
		scale = 1.0f; 
		panels = Panels::thePanels;
	if(swapAmbients) {
		interpOwner = true;
		interps = new Interpolators(); 
	} else {
		interpOwner = false;
		interps = MasterController::curAmbient->interps;
	}
}


Ambient::~Ambient () { 
	if(interpOwner) {
	  delete interps; 
	}
  interps = NULL;
}

AmbientPixel* Ambient::addAmbientPixel(int panelName, int c, int r, int stick) {
    Panel *panel = &(Panels::thePanels->panels[panelName]);
      return new AmbientPixel(panel, c, r, stick);  // neg valus will be whole light
}

void Ambient::update(WorldStats *worldStats, int ct, int dt, float curScale) {
  if(interps == NULL) return;
  interps->update(ct, dt, curScale);
  updateFrame(worldStats, ct, dt, interps);
}

void Ambient::addColor(int panel, int col, int row, int r, int g, int b) {
  panels[panel].getPixel(col, row)->addColor(r * scale, g * scale, b * scale);
}
void Ambient::addColor(int panel, int col, int row, int stick, int r, int g, int b) {
  panels[panel].getPixel(col, row)->getSubPixel(stick)->addColor(r * scale, g * scale, b * scale);
}

//void Ambient::addPixel(int panelName, int c, int r) {
//}
