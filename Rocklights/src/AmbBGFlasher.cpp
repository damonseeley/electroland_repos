#include "AmbBGFlasher.h"


AmbBGFlasher::AmbBGFlasher(bool ci) : Ambient(ci) {
	string setting = MasterController::curMasterController->name + "FlashProb";
	flashProb = CProfile::theProfile->Float(setting.c_str(), .5);
	setting = MasterController::curMasterController->name + "FlashAttemps";
	flashAttempPerFrame = CProfile::theProfile->Int(setting.c_str(), 1);
}
void AmbBGFlasher::flash(int p, int c, int r) {
			BasePixel *px = Panels::thePanels->getPixel(p, c, r);
			if(! px->isTarget) {
				if(random(100) > 50) {
					px->addColor(random(4), 0, 255, 0);
				} else {
					px->addColor(random(4), 0, 0, 255);
				}
	
			}
}

void AmbBGFlasher::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	for(int i = 0; i < flashAttempPerFrame; i++) {
	if((((float) random(100)) / 100.0f) < flashProb) {
		if(random(100) > 30) {
			int c = random(Panels::thePanels->getWidth(Panels::A));
			int r = random(Panels::thePanels->getHeight(Panels::A));
			flash(Panels::A, c, r);
		} else {
			if(random(18) > 7) {
				flash(random(8)+2, random(7), 0);
			} else {
				flash(Panels::B, random(7), random(6));
			}
		}
	}
	}
}
