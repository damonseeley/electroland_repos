#include "AmbTron.h"
#include "Panels.h"
#include "StringUtils.h"

// for pillars always have row = 0

// format Panel, col, row, isTarget, ..... -1
// (tagets pause for retraction)
/*
int AmbTron::path[] = { 
	Panels::A, 0, 1, 0,
		Panels::A, 4, 1, 0,
		Panels::A, 4, 3, 0,
		Panels::A, 15, 9, 1,
		Panels::C, 1, 0, 1,
		Panels::B, 2, 1, 1,
		Panels::B, 5, 4, 1,
		Panels::A, 9, 1, 1,
		Panels::I, 5, 0, 1,
		Panels::A, 14, 2, 0,
		Panels::E, 5, 0, 1,
		Panels::A, 12, 5, 1,
		Panels::J, 2, 0, 1,
		Panels::A, 6, 8, 1,
		Panels::A, 18, 4, 1,
		Panels::A, 6, 3, 0,
		Panels::A, 6, 1, 0,
		Panels::A, 0, 1, 1,
		-1
};

*/

AmbTron::AmbTron(bool ci) : Ambient(ci) {
	isRunning = true;

	string setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() + "TronMSPerSquare";
//	cout << "setting " << setting << endl;
	msPerSquare = CProfile::theProfile->Int(setting.c_str(), 5);
	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() +"TronR";
	r = CProfile::theProfile->Int(setting.c_str(), 255);
	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() +"TronG";
	g = CProfile::theProfile->Int(setting.c_str(), 0);
	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() +"TronB";
	b = CProfile::theProfile->Int(setting.c_str(), 0);

	setting = MasterController::curMasterController->name +MasterController::curMasterController->composerId.str() + "TronPath";
	string str = CProfile::theProfile->String(setting.c_str(), "");
	vector<string> ls;
	StringUtils::split(str, ls);
	path = StringUtils::createIntArray(ls);

	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() +"TronSounds";
	str = CProfile::theProfile->String(setting.c_str(), "");
	snds = new StringSeq(setting, "NONE");
		

	curIndex = 8;
	curP = path[0];
	curC = path[1];
	curR = path[2];
	// ignore parth[3]
	gP = path[4];
	gC = path[5];
	gR = path[6];
	gIsPause = (path[7] == 1);
	points.push_back(Panels::thePanels->getPixel(curP, curC, curR));
	timeLeft = msPerSquare;

	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() + "TronTailLength";
	tailLength = CProfile::theProfile->Int(setting.c_str(), -1);

	setting = MasterController::curMasterController->name + MasterController::curMasterController->composerId.str() + "TronStabLength";
	stabLength = CProfile::theProfile->Int(setting.c_str(), -1);
	timeLeftForStab =stabLength;
	expanding = true;

}
AmbTron::~AmbTron() {
	stabPoints.clear();
	delete[] path;
	delete snds;
}

bool AmbTron::atLoc(int p, int c, int r) {
	return (curP == p) && (curC == c) &&
		(curR == r);
}
void AmbTron::moveToLoc(int p, int c, int r) {

	if(curP == p) {
		if(curC > c) {
			curC--;
		} else if(curC < c) {
			curC++;
		} else if (curR > r) {
			curR--;
		} else if (curR < r) {
			curR++;
		}
	} else {
		if(curP == Panels::A) {
			switch(gP) {
						case Panels::B:
							if(atLoc(Panels::A, 2, gR+3)) {
								curP = Panels::B;
								curC =  0;
								curR = gR;
							} else {
								moveToLoc(Panels::A, 2, gR+3);
							}
							break;
						case Panels::C:
							if(atLoc(Panels::A, 5, 11)) {
								curP = Panels::C;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 5, 11);
							}
							break;
						case Panels::D:
							if(atLoc(Panels::A, 12, 11)) {
								curP = Panels::D;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 12, 11);
							}
							break;
						case Panels::E:
							if(atLoc(Panels::A, 19, 10)) {
								curP = Panels::E;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 19, 10);
							}
							break;
						case Panels::F:
							if(atLoc(Panels::A, 17, 11)) {
								curP = Panels::F;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 17, 11);
							}

							break;
						case Panels::G:
							if(atLoc(Panels::A, 17, 0)) {
								curP = Panels::G;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 17, 0);
							}

							break;
						case Panels::H:
							if(atLoc(Panels::A, 19, 1)) {
								curP = Panels::H;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 19, 1);
							}
							break;
						case Panels::I:
							if(atLoc(Panels::A, 12, 0)) {
								curP = Panels::I;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 12, 0);
							}
							break;
						case Panels::J:							
							if(atLoc(Panels::A, 5, 0)) {
								curP = Panels::J;
								curC = 0;
								curR = 0;
							} else {
								moveToLoc(Panels::A, 5, 0);
							}

							break;
			}
		} else {
			// on pillar
			if(curC == 0) { 
				// transition to ceiling
				switch(curP) {
						case Panels::B:
							curC = 2;
							curR = curR + 3;
							break;
						case Panels::C:
							curC = 5;
							curR = 11;
							break;
						case Panels::D:
							curC = 12;
							curR = 11;
							break;
						case Panels::E:
							curC = 19;
							curR = 10;
							break;
						case Panels::F:
							curC = 17;
							curR = 11;
							break;
						case Panels::G:
							curC = 17;
							curR = 0;
							break;
						case Panels::H:
							curC = 19;
							curR = 1;
							break;
						case Panels::I:
							curC = 12;
							curR = 0;
							break;
						case Panels::J:
							curC = 5;
							curR = 0;
							break;
				}
				curP = Panels::A;

			} else{
				curC--;
			}

		}
	}
	points.push_back(Panels::thePanels->getPixel(curP, curC, curR));
}

void AmbTron::advanceGoalPoint() {
	//				curIndex += 4;
	curP = gP;
	gP = path[curIndex++];
	if(gP != -1) {
		curC = gC;
		curR = gR;
		gC = path[curIndex++];
		gR = path[curIndex++];
		gIsPause = (path[curIndex++] == 1);
		BasePixel *pix = Panels::thePanels->getPixel(curP, curC, curR);
		if(gIsPause && (stabLength > 0)) {
			stabPoints.push_back(pix);
		}
		points.push_back(pix);
		expanding = true;
	} else {
		isRunning = false;
		if(stabLength <= 0) {
			if(caller != NULL) {
				caller->timeToIntermission = 0;
			}
		}
	}
}

void AmbTron::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
	if(isRunning) {
		if(timeLeft <= 0) {
			if(expanding) {
				if(atLoc(gP, gC, gR)) {
					if(gIsPause) {
						expanding = false; // start contracting
						string snd = snds->getEl();
						if((snd != "") && (snd != "NONE")) { 
							SoundHash::theSoundHash->play(snds->getEl());
						}
					} else {
						advanceGoalPoint();
					}
				} else {
					moveToLoc(gP, gC, gR);
				}
			} else {
				points.pop_front();
				if(points.size() == 0) { // done contracting
					advanceGoalPoint();
				}
			}
			timeLeft = msPerSquare;
		} else {
			timeLeft -= dt;
		}

		if(tailLength > 0) {
			while(points.size() > (unsigned) tailLength) {
				points.pop_front();
			}
		}

		list<BasePixel *>::iterator iter;
		for(iter = points.begin(); iter != points.end(); iter++) {
			(*iter)->addColor(r,g,b);
		}
	} else {
		if(stabLength > 0) {
			if(timeLeftForStab <= 0) {
				if(caller != NULL) {
					caller->timeToIntermission = 0;
				}
				stabLength = -1;
			} else {
				for(unsigned int i = 0; i < stabPoints.size(); i++) {
					stabPoints[i]->addColor(r,g,b);
				}
				timeLeftForStab-=dt;
			}
		}
	}
}
;