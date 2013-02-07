#include "ambcomposer.h"
#include "StringUtils.h"
#include "AmbientCreatorHash.h"


AmbComposer::AmbComposer(bool ci) : Ambient(ci)
{
	string setting = MasterController::curMasterController->name + "Ambients";
	string str = CProfile::theProfile->String(setting.c_str(), "");
	
	setting = MasterController::curMasterController->name + "CompDelay";
	compDelay = CProfile::theProfile->Int(setting.c_str(), -1);
	StringUtils::split(str, ambNameLs);
	MasterController::curAmbient = this; // need this so composed ambients can get to interps
	if (compDelay <= 0) {
		while(ambNameLs.size() > 0) {

			MasterController::curMasterController->composerId.str("");
			MasterController::curMasterController->composerId << (ambNameLs.size() -1);
//			cout << "id is" << MasterController::curMasterController->composerId.str() << endl;
			ambVec.push_back(AmbientCreatorHash::theACHash->create(ambNameLs.back(), false));
			ambNameLs.pop_back();
		}
		timeLeft = INT_MAX;
	} else {
			MasterController::curMasterController->composerId.str("");
			MasterController::curMasterController->composerId << (ambNameLs.size() -1);
		ambVec.push_back(AmbientCreatorHash::theACHash->create(ambNameLs.back(), false));
		ambNameLs.pop_back();
		timeLeft = compDelay;
	}
}

AmbComposer::~AmbComposer(void)
{

	while(ambVec.size() > 0) {
		Ambient *a = ambVec.back();
		ambVec.pop_back();
		delete a;
	}
}

void AmbComposer::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps)  {
	if(compDelay > 0) {
		if(timeLeft <= 0) {
			if(ambNameLs.size() > 0) {
				MasterController::curMasterController->composerId.str("");
			MasterController::curMasterController->composerId << (ambNameLs.size() -1);
				ambVec.push_back(AmbientCreatorHash::theACHash->create(ambNameLs.back(),false));
				ambNameLs.pop_back();
				timeLeft = compDelay;
			} else {
				compDelay = -1;
			}
		} else {
			timeLeft -= dt;
		}
	}
	for(unsigned int i = 0; i < ambVec.size(); i++) {
		ambVec[i]->updateFrame(worldStats, ct, dt, interps);
	}
}

