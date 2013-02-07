#include "Pattern1Square.h"


Pattern1Square::Pattern1Square(int aveMode, int tail, int dir) : Pattern() {
  mode = aveMode;
//  tailDelay = tail;
  pilDir = dir;
}

void Pattern1Square::setAvatars(PersonStats *ps, bool isEnter) {
  //  co ut << "Pattern1Square::setAvatars with grp " << PeopleStats::curAvatarGroup << endl;
//	Avatar *a = new AV1Square(ps, ps->inpterps);
	Avatar *a = AvatarCreatorHash::theACHash->create("1Square", ps, ps->inpterps);
  //UNDONE CHANGE TO VARS
  a->setPillarMode(pilDir);
//  a->setTrailDelay(tailDelay);
  
  ps->addAvatar(a, PeopleStats::curAvatarGroup);
}