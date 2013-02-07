#include "AmbRedStick.h"
#include "MasterController.h"

int AmbRedStick::red1[] = { 
  0,    0, 0,   0, 0, 0,  10000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};

int AmbRedStick::red2[] = { 
  0,    0, 0,   0, 0, 0,  18000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};
int AmbRedStick::red3[] = { 
  0,    0, 0,   0, 0, 0,  28000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};

int AmbRedStick::red4[] = { 
  0,    0, 0,   0, 0, 0,  38000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};

int AmbRedStick::red1T[] = { 
  0,    0, 0,   0, 0, 0,  10000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};

int AmbRedStick::red2T[] = { 
  0,    0, 0,   0, 0, 0,  18000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};
int AmbRedStick::red3T[] = { 
  0,    0, 0,   0, 0, 0,  28000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};

int AmbRedStick::red4T[] = { 
  0,    0, 0,   0, 0, 0,  38000,
    0, 0, 0, 255, 0, 0, 2000,
    255, 0, 0, 0, 0, 0, 2000,
    -1
};


//float AmbRedStick::densityFactor = -1.0f;
//float AmbRedStick::targetDensityFactor = -1.0f;

AmbRedStick::AmbRedStick(bool ci) : Ambient(ci) {
	  string setting = MasterController::curMasterController->name + "RedStickDensityScalor";
	  densityFactor =  CProfile::theProfile->Float(setting.c_str(), 1.0f);
	setting = MasterController::curMasterController->name + "TrargetDensityScalor";
	targetDensityFactor = CProfile::theProfile->Float(setting.c_str(), 1.0f);

	setting = MasterController::curMasterController->name + "RedStickSpeedScalor";
	speedFactor = CProfile::theProfile->Float(setting.c_str(), 1.0f);

    red1[6] = 1000.0f  * densityFactor;
    red2[6] = 1800.0f  * densityFactor;
    red3[6] = 2800.0f  * densityFactor;
    red4[6] = 3800.0f  * densityFactor;
    red1T[6] = 1000.0f  *targetDensityFactor ;
    red2T[6] = 1800.0f  *targetDensityFactor ;
    red3T[6] = 2800.0f  *targetDensityFactor ;
    red4T[6] = 3800.0f  *targetDensityFactor ;


  
/*
for(int i = 0; i < 21; i++) {
int row = i * 7;
seas[row + 1] *= 4;
seas[row + 2] *= 4;
seas[row + 4] *= 4;
seas[row + 5] *= 4;
}
  */
  for(int p = 0; p < Panels::PANEL_CNT; p++) {
    
    for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
      for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
        for(int s = 0; s < 4; s++) { 
          float frac = ((float) random(100) / 100.0f);          
          if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
            int rnd = random(4);
            if (rnd == 0) {

              (new IGeneric(interps, addAmbientPixel(p, c, r, s), red1, -1, frac))->timeScale = speedFactor;
            } else if (rnd == 1) {
              (new IGeneric(interps, addAmbientPixel(p, c, r, s), red2, -1, frac))->timeScale = speedFactor;
            } else if (rnd == 2) {
              (new IGeneric(interps, addAmbientPixel(p, c, r, s), red3, -1, frac))->timeScale = speedFactor;
            } else if (rnd == 3) {
              (new IGeneric(interps, addAmbientPixel(p, c, r, s), red4, -1, frac))->timeScale = speedFactor;
            }
          } else {
            if (s == 0) {
              int rnd = random(4);
              if (rnd == 0) {
                (new IGeneric(interps, addAmbientPixel(p, c, r), red1T, -1, frac))->timeScale = speedFactor;
              } else if (rnd == 1) {
                (new IGeneric(interps, addAmbientPixel(p, c, r), red2T, -1, frac))->timeScale = speedFactor;
              } else if (rnd == 2) {
                (new IGeneric(interps, addAmbientPixel(p, c, r), red3T, -1, frac))->timeScale = speedFactor;
              } else if (rnd == 3) {
                (new IGeneric(interps, addAmbientPixel(p, c, r), red4T, -1, frac))->timeScale = speedFactor;
              }
              
              //              new IGeneric(interps, addAmbientPixel(p, c, r, s), red, -1, frac);
            }
          }
        }
      }
    }
  }
}

void AmbRedStick::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
}
