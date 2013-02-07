#include "PatGeneric.h"
#include "StringUtils.h"

PatGeneric::PatGeneric(string s) : Pattern() {
	name = s;
	CProfile *prof = CProfile::theProfile;
	string setting = name + "Avatar";
	string aveName = prof->String(setting.c_str(), "defaultAvatar");
	if (aveName == "NONE") {
		useAvatar = false;
	} else {
		useAvatar = true;
		vector<string> tmpls;
		StringUtils::split(aveName, tmpls);
		for(unsigned int i = 0; i < tmpls.size(); i++) {
			AvatarCreator *a = AvatarCreatorHash::theACHash->get(tmpls[i]);
			if(a != NULL) {
				avatarCreatorls.push_back(a);
			} else {
						timeStamp(); clog << "WARNING  Attempt to use undefined Avatar name " << tmpls[i] << "\n";
			}
		}
		tmpls.clear();
		if (avatarCreatorls.size() == 0) {
			avatarCreatorls.push_back(AvatarCreatorHash::theACHash->defaultCreator);
		}
		setting = name + "EnterPat";
		string arg = prof->String(setting.c_str(), "NONE");
		if(arg != "NONE") {
			StringUtils::split(arg, tmpls);
			for(unsigned int i = 0; i < tmpls.size(); i++) {
				Arrangement *a = ArrangementHash::theHash->get(tmpls[i]);
				if(a == NULL) {
						timeStamp(); clog << "WARNING  Attempt to use undefined EnterPat Arrangement name " << tmpls[i] << " in " << name << "\n";
				}
				enterArrangementls.push_back(a);
			}
			tmpls.clear();
		} else {
			enterArrangementls.push_back(NULL);
		}
		setting = name + "ExitPat";
		arg = prof->String(setting.c_str(), "NONE");
		if(arg != "NONE") {
			StringUtils::split(arg, tmpls);
			for(unsigned int i = 0; i < tmpls.size(); i++) {
				Arrangement *a = ArrangementHash::theHash->get(tmpls[i]);
								if(a == NULL) {
						timeStamp(); clog << "WARNING  Attempt to use undefined EnterPat Arrangement name " << tmpls[i] << " in " << name << "\n";
				}

				exitArrangementls.push_back(a);
			}
			tmpls.clear();
		} else {
			exitArrangementls.push_back(NULL);
		}

		setting = name + "MovePat";
		arg = prof->String(setting.c_str(), "NONE");
		if(arg != "NONE") {
			StringUtils::split(arg, tmpls);
			for(unsigned int i = 0; i < tmpls.size(); i++) {
				Arrangement *a = ArrangementHash::theHash->get(tmpls[i]);
												if(a == NULL) {
						timeStamp(); clog << "WARNING  Attempt to use undefined EnterPat Arrangement name " << tmpls[i] << " in " << name << "\n";
				}

				moveArrangementls.push_back(a);
			}
			tmpls.clear();
		} else {
			moveArrangementls.push_back(NULL);
		}

		setting = name + "OverheadPat";
		arg = prof->String(setting.c_str(), "NONE");
		if(arg != "NONE") {
			StringUtils::split(arg, tmpls);
			for(unsigned int i = 0; i < tmpls.size(); i++) {
				Arrangement *a = ArrangementHash::theHash->get(tmpls[i]);
				if(a == NULL) {
						timeStamp(); clog << "WARNING  Attempt to use undefined OverheadPat Arrangement name " << tmpls[i] << " in " << name << "\n";
				}

				overheadArrangementls.push_back(a);
			}
			tmpls.clear();
		} else {
			overheadArrangementls.push_back(NULL);
		}

		setting = name + "EnterSound";
		StringUtils::split(prof->String(setting.c_str(), ""), enterSoundls);
		setting = name + "ExitSound";
		StringUtils::split(prof->String(setting.c_str(), ""), exitSoundls);
		setting = name + "MoveSound";
		StringUtils::split(prof->String(setting.c_str(), ""), moveSoundls);

		setting = name + "EnterSoundLoop";
		enterSoundLoop = prof->Int(setting.c_str(), 1);

		setting = name + "TailDelay";
		tailDelay = prof->Int(setting.c_str(), 200);

		setting = name + "PillarMode";
		pillarMode = prof->Int(setting.c_str(), 1);

		setting = name + "AmbientSound";
		StringUtils::split(prof->String(setting.c_str(), ""), ambientSoundls);
		setting = name + "AmbientSoundLoop";
		ambientSoundLoop = prof->Int(setting.c_str(), 1);
	}
		if(! ambientSoundls.empty()) {
			if (ambientSoundLoop <= 0) {
				//TODO change this to seq
				ambientSoundLoopNumber = SoundHash::theSoundHash->play(ambientSoundls[random(ambientSoundls.size())], ambientSoundLoop);
			} else {
				SoundHash::theSoundHash->play(ambientSoundls[random(ambientSoundls.size())], ambientSoundLoop);
				ambientSoundLoopNumber = -1;
			}
		} else {
			ambientSoundLoopNumber = -1;
		}



}

PatGeneric::~PatGeneric(){
	if (ambientSoundLoopNumber != -1) {
		string setting = name + "AmbientSoundStopImmediate";
		SoundHash::theSoundHash->stop(ambientSoundLoopNumber, CProfile::theProfile->Bool(setting.c_str(), false));
	}
}

void PatGeneric::setAvatars(PersonStats *ps, bool isEnter) {
	if (useAvatar) {
		int sz = avatarCreatorls.size();
		if(sz == 0) return;
		AvatarCreator *ac = avatarCreatorls[random(sz)];
		Avatar *a = ac->create(ps, ps->inpterps);
			a->setColor(ps->color);
		if(ac->type == AvatarCreator::GENERIC) {
			a->setName(name);
			a->setEnterArrangement(enterArrangementls[random(enterArrangementls.size())]);
			a->setExitArrangement(exitArrangementls[random(exitArrangementls.size())]);
			a->setMoveArrangement(moveArrangementls[random(moveArrangementls.size())]);
			a->setOverheadArrangement(overheadArrangementls[random(overheadArrangementls.size())]);
			a->setEnterSound(enterSoundls[random(enterSoundls.size())]);
			a->setExitSound(exitSoundls[random(exitSoundls.size())]);
			a->setMoveSound(moveSoundls[random(moveSoundls.size())]);
			a->setEnterSoundLoop(enterSoundLoop);
			a->setTrailDelay(tailDelay);
			a->setPillarMode(pillarMode);
			a->init(ps->inpterps);
			if(isEnter)
				a->unEntered = true;
		}
		ps->addAvatar(a, PeopleStats::curAvatarGroup);
	}
}