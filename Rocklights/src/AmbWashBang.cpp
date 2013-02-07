#include "ambwashbang.h"

int AmbWashBang::wash[] = { 
	 0, 0, 0, 0, 0, 0, 7000,
	 0, 0, 0, 255, 0, 0, 2000,
	 255, 0, 0, 255, 0, 0, INT_MAX,
    -1
};

AmbWashBang::AmbWashBang(bool ci) : Ambient(ci)
{
		string setting = MasterController::curMasterController->name + "WashSpeedScale";
		washSpeedScale = CProfile::theProfile->Float(setting.c_str(), 1.0f);
		int r;
		for(r = 0; r< 7; r++) {
		for(int c = 0; c < 7; c++) {
			for(int s = 0; s < 4; s++) {
				BasePixel *px = Panels::thePanels->getPixel(Panels::B, c, r);
				if(! px->isTarget) {
					(new IGeneric(interps, addAmbientPixel(Panels::B, c, r, s), wash, -1, ((float) (c*4)+s) / 28.0f))->timeScale = washSpeedScale;
				} else if (s == 3) {
					(new IGeneric(interps, addAmbientPixel(Panels::B, c, r), wash, -1, ((float) (c*4)+s) / 28.0f))->timeScale = washSpeedScale;
				}
			}
		}
		}

		for(int p = Panels::C; p <= Panels::J; p++) {
			for(int c = 0; c < 7; c++) {
				for(int s = 0; s < 4; s++) {
				BasePixel *px = Panels::thePanels->getPixel(p, c, r);
				if(! px->isTarget) {
					(new IGeneric(interps, addAmbientPixel(p, c, 0, s), wash, -1, ((float) (c*4)+s) / 28.0f))->timeScale = washSpeedScale;
				} else if (s== 3) {
					(new IGeneric(interps, addAmbientPixel(p, c, 0), wash, -1, ((float) (c*4)+s) / 28.0f))->timeScale = washSpeedScale;
				}
			}
		}
		}
		setting = MasterController::curMasterController->name + "WashCutShort";
		timeToStab = (((float)(wash[6] + wash[13])) / washSpeedScale) -
			CProfile::theProfile->Int(setting.c_str(), 0);


}

void AmbWashBang::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	timeToStab-=dt;
	if(timeToStab <= 0) {
		for(int c = 0; c < Panels::thePanels->getWidth(Panels::A); c++) {
			for(int r = 0; r < Panels::thePanels->getHeight(Panels::A); r++) {
				BasePixel *px = Panels::thePanels->getPixel(Panels::A, c, r);
				if(px->isTarget) {
					px->addColor(random(255), 0,0); 
				} else {
					px->addColor(0, random(255), random(255), random(255));
					px->addColor(1, random(255), random(255), random(255));
					px->addColor(2, random(255), random(255), random(255));
					px->addColor(3, random(255), random(255), random(255));
				}

			}
		}
	}
}

AmbWashBang::~AmbWashBang(void)
{
}
