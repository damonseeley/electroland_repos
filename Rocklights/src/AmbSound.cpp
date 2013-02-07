#include "AmbSound.h"
#include "StringUtils.h"
#include "SoundHash.h"
AmbSound::AmbSound(bool ci) : Ambient(ci) { 
	string setting = MasterController::curMasterController->name +  "SoundCues";
	string str = CProfile::theProfile->String(setting.c_str(), "");
	vector<string> ls;
	StringUtils::split(str, ls);
	for(unsigned int i = 0; i < ls.size(); i+=2) {
		SoundCue *c = new SoundCue(atoi(ls[i].c_str()), ls[i+1]);
		cues.push_back(c);
	}
	
	if(! cues.empty()) {
		list<SoundCue *>::iterator iter;
		iter = cues.begin();
		SoundCue *c = (*iter);
		timeLeft = c->delayBeforeSound;
		nextSound = c->snd;
		delete c;
		cues.pop_front();
	}
}
AmbSound::~AmbSound(){
		list<SoundCue *>::iterator iter;
	for(iter = cues.begin(); iter != cues.end(); iter++) {
		SoundCue *c = (*iter);
		delete c;
	}
	cues.clear();

}
void AmbSound::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	timeLeft-=dt;
	if(timeLeft <= 0) {
		if(nextSound != "") {
			SoundHash::theSoundHash->play(nextSound);
		}
		if(! cues.empty()) {
			list<SoundCue *>::iterator iter;
			iter = cues.begin();
			SoundCue *c = (*iter);
			timeLeft = c->delayBeforeSound;
			nextSound = c->snd;
			delete c;
			cues.pop_front();
		} else {
			timeLeft = 60000;
		}
	}
	
	}
