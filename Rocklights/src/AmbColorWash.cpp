#include "AmbColorWash.h"
#include "MasterController.h"
#include "profile.h"
#include <vector>
#include <string>
#include "StringUtils.h"

AmbColorWash::AmbColorWash(bool ci) : Ambient(ci) {
	string setting = MasterController::curMasterController->name + "ColorWash";
	vector<string> tmp;
	StringUtils::split(CProfile::theProfile->String(setting.c_str(),"0,0,0,0,0,0,1000,-1"), tmp);
	setting = MasterController::curMasterController->name + "ColorWashLoop";
	bool loop = CProfile::theProfile->Bool(setting.c_str(), false);
	int loopNum;
	if(! loop){
		tmp.push_back("0");
		tmp.push_back("0");
		tmp.push_back("0");
		tmp.push_back("0");
		tmp.push_back("0");
		tmp.push_back("0");
		tmp.push_back("3600000");
		tmp.push_back("-1");
		loopNum = 1;
	} else {
		loopNum = -1;
		tmp.push_back("-1");
	}
	colorIntp = StringUtils::createIntArray(tmp);
	
	

  for(int p = 0; p < Panels::PANEL_CNT; p++) {    
    for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
      for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
        for(int s = 0; s < 4; s++) { 
          float frac = ((float) random(100) / 100.0f);          
          if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
			  new IGeneric(interps, addAmbientPixel(p, c, r, s), colorIntp,loopNum);
          } else {
            if (s == 0) {
				new IGeneric(interps, addAmbientPixel(p, c, r), colorIntp,loopNum);
			}
		  }
		}
	  }
	}
  }
}

void AmbColorWash::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
}
