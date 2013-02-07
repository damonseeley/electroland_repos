#include "ambblinktag.h"
#include <string>
#include "Profile.h"
#include "PeopleStats.h"
#include "PersonStats.h"
#include "SoundHash.h"

#define BPLUS 1
#define BX 2
#define BBOX 3


AmbBlinkTag::AmbBlinkTag(bool ci): Ambient(ci) {
	string setting = MasterController::curMasterController->name+"BlinkOnTime";
	blinkOnTime = CProfile::theProfile->Int(setting.c_str(), 500);
	
	setting = MasterController::curMasterController->name+"BlinkOffTime";
	blinkOffTime = CProfile::theProfile->Int(setting.c_str(), 500);
	
	 setting = MasterController::curMasterController->name+"BlinkTimes";
	blinkTimes = CProfile::theProfile->Int(setting.c_str(), 5);

	setting = MasterController::curMasterController->name+"IntraBlinkDelay";
	intraBlinkDelay = CProfile::theProfile->Int(setting.c_str(), 1000);

	setting = MasterController::curMasterController->name+"BlinkSounds";
	sounds = new StringSeq(setting, "");

	setting = MasterController::curMasterController->name+"BlinkStyle";
	setting = CProfile::theProfile->String(setting.c_str(), "");
	if(setting == "plus") {
		style = BPLUS;
	} else if (setting == "X") {
		style = BX;
	} else if (setting == "box") {
		style  = BBOX;
	} else {
		style = 0;
	}

	reset();
}
void AmbBlinkTag::reset() {
	timeLeft = blinkOnTime;
	blinksLeft =blinkTimes;
	state = ON;

	PersonStats *curPerson = PeopleStats::thePeopleStats->getRandom();
	if(curPerson!=NULL) {
		curId = curPerson->id;
	} else {
		timeLeft = 0;
		state = DELAY;
		cout << "its null" << endl;
	}
	SoundHash::theSoundHash->play(sounds->getEl());

}
void AmbBlinkTag::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	if(timeLeft <= 0) {
		switch(state) {
			case ON:
				blinksLeft--;
				if(blinksLeft <= 0) {
					state = DELAY;
					timeLeft = intraBlinkDelay;
				} else {
					state = OFF;
					timeLeft = blinkOffTime;
				}
				break;
			case OFF:
				timeLeft = blinkOnTime;
				state = ON;
				break;
			case DELAY:
				reset();
				break;
		}
	} else {
		if(state == ON) {
			PersonStats *curPerson  = PeopleStats::thePeopleStats->get(curId);
			if(curPerson == NULL) { // person exited
				state = DELAY;
				timeLeft = intraBlinkDelay;
			} else {
				Panels::thePanels->getPixel(Panels::A, curPerson->col, curPerson->row)->addColor(255,0,0); 
				switch(style) {
					case BBOX:
				Panels::thePanels->getPixel(Panels::A, curPerson->col-1, curPerson->row-1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col+1, curPerson->row-1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col-1, curPerson->row+1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col+1, curPerson->row+1)->addColor(255,0,0); 

						// no break on purpose
					case BPLUS:
				Panels::thePanels->getPixel(Panels::A, curPerson->col-1, curPerson->row)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col+1, curPerson->row)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col, curPerson->row+1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col, curPerson->row-1)->addColor(255,0,0); 
				break;

					case BX:
										Panels::thePanels->getPixel(Panels::A, curPerson->col-1, curPerson->row-1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col+1, curPerson->row-1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col-1, curPerson->row+1)->addColor(255,0,0); 
				Panels::thePanels->getPixel(Panels::A, curPerson->col+1, curPerson->row+1)->addColor(255,0,0); 
					

				}
			}
		}
		timeLeft -= dt;
	}
}

AmbBlinkTag::~AmbBlinkTag()
{
}
