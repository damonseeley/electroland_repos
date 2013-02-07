#include "AgmtWashEnter.h"
#include "Agmt1Square.h"
#include "MasterController.h"

int AgmtWashEnter::red[] = { 255, 0,0, 0,0,0, 1000, -1};
int AgmtWashEnter::green[] = { 0, 255,0, 0,0,0, 1000, -1};
int AgmtWashEnter::blue[] = { 0, 0,255, 0,0,0, 1000, -1};

	 int AgmtWashEnter::intensity = 255;
	 int AgmtWashEnter::fadeTime = 1000;



AgmtWashEnter::AgmtWashEnter() {
	string setting = "WashEnterIntensity";
	intensity =  CProfile::theProfile->Int(setting.c_str(), 255);
	
	setting = "WashEnterFadeTime";
	fadeTime =  CProfile::theProfile->Int(setting.c_str(), 1000);	

	red[0] = intensity;
	green[0] = intensity;
	blue[0] = intensity;

	red[6] = fadeTime;
	green[6] = fadeTime;
	blue[6] = fadeTime;

}


void AgmtWashEnter::apply(Avatar *avatar, Interpolators *interps) {

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
	//avatar->setColor(avatar->getColor());
	for(int p = 0; p < Panels::PANEL_CNT; p++) {    
		Panel *panel = &(Panels::thePanels->panels[p]);
		for (int c = 0; c < panel->getWidth(); c++) {
			for(int r = 0; r < panel->getHeight(); r++) {
				if(panel->getPixel(c,r)->isTarget) {
					new IGeneric(interps,  new AmbientPixel(panel, c, r, -1), colorIntp);
				} else {
					for(int s = 0; s < 4; s++) { 
						new IGeneric(interps, new AmbientPixel(panel, c, r, s), colorIntp);
					}
				}
			}
		}
	}
}
