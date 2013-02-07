#include "MCGeneric.h"
#include "StringUtils.h"
#include "MCPause.h"



MCGeneric::MCGeneric(PeopleStats *peopleStats, string mcName, string mcTransFromName) : MasterController(peopleStats) {
	name = mcName;
	transFromName=mcTransFromName;

	string setting = name + "LowThresh";
	lowTresh = CProfile::theProfile->Int(setting.c_str(), -1);

	setting = name + "LowTransToName";
	lowTransTols = new StringSeq(setting, "defaultState");

	setting = name + "HighThresh";
	highThresh = CProfile::theProfile->Int(setting.c_str(), -1);

	setting = name + "HighTransToName";
	highTransTols = new StringSeq(setting, "defaultState");

	setting = name + "LifeTime";
	lifeTime = CProfile::theProfile->Int(setting.c_str(), -1);
	timeToIntermission = lifeTime;

	setting = name + "TimeTransToName";
	timeTransTols = new StringSeq(setting, "defaultState");


	setting = name + "Ambient";
	StringSeq *tmpLs =  new StringSeq(setting, "defaultState");
	ambientName = tmpLs->getEl();
	delete tmpLs;

	timeStamp(); clog << "INFO  entering State:" << name << "\n";

	if (ambientName == "NONE") {
		timeStamp(); clog << "INFO         Ambient:AmbEmpty" << "\n";
		curAmbient = AmbientCreatorHash::theACHash->create("AmbEmpty");
		curAmbient->setCaller(this);
	} else {
		timeStamp(); clog << "INFO         Ambient:" << ambientName << "\n";
		curAmbient = AmbientCreatorHash::theACHash->create(ambientName);
		curAmbient->setCaller(this);
	}

	setting = name + "TransSpeed";
	// this is a mess
	int transSpeed = CProfile::theProfile->Int(setting.c_str(), MasterController::globalTransitionCrossFadeTime);
	MasterController::transitionCrossFadeTime = transSpeed;
	peopleStats->transitionTo(new PatGeneric(name), transSpeed);

	setting = name + "AfterTimePause";
	afterPauseTime = CProfile::theProfile->Int(setting.c_str(), -1);

}




void MCGeneric::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
//	cout << name << " " << isInTransition << endl;
  if(! isInTransition) {
	  if(lowTresh >= 0) {
	    if(worldStats->peopleCnt < lowTresh) {
			string newState = lowTransTols->getEl();
			if(newState == "BACK") {
				newState = transFromName;
			}
			new MCGeneric(ps, newState, name);
			return;
		}
	  }
//	  cout << "        " << worldStats->peopleCnt << " >? " << highThresh << endl;
	  if(highThresh >= 0) {
	    if(worldStats->peopleCnt > highThresh) {
			string newState = highTransTols->getEl();
			if(newState == "BACK") {
				newState = transFromName;
			}
			new MCGeneric(ps, newState, name);
			return;
		}
	  }

	  if(lifeTime >= 0) {
		  timeToIntermission -= dt;
		  if(timeToIntermission <= 0) {
			string newState = timeTransTols->getEl();
			if(newState == "BACK") {
				newState = transFromName;
			}
			if(afterPauseTime > 0) {
				new MCPause(ps, newState, name, afterPauseTime);
			} else {
				new MCGeneric(ps, newState, name);
			}
			return;
		  }
	  }
  }
}

  


MCGeneric::~MCGeneric() {
	delete lowTransTols;
	delete highTransTols;
	delete timeTransTols;

//	cout<<" MCGeneric destroyed" << endl;
}
