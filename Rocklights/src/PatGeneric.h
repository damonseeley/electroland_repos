#ifndef __PATGENERIC_H__
#define __PATGENERIC_H__

#include <string>
#include <vector>
#include "Pattern.h"
#include "AvatarCreatorHash.h"
#include "AvatarCreator.h"
#include "ArrangementHash.h"

class PatGeneric : public Pattern {
	vector<AvatarCreator *> avatarCreatorls;
	bool useAvatar;

	string name;
  vector<Arrangement*> enterArrangementls;
  vector<Arrangement*> exitArrangementls;
  vector<Arrangement*> moveArrangementls;
  vector<Arrangement*> overheadArrangementls;
  vector<string> enterSoundls;
  vector<string> exitSoundls;
  vector<string> moveSoundls;

  vector<string> ambientSoundls;
  int ambientSoundLoopNumber;
  int ambientSoundLoop;

  int enterSoundLoop;
  int tailDelay;
  int pillarMode;


public:

  PatGeneric(string name);
    virtual ~PatGeneric();

  virtual void setAvatars(PersonStats *ps, bool isEnter);

}
;

#endif