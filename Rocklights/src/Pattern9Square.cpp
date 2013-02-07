#include "Pattern9Square.h"

void Pattern9Square::setAvatars(PersonStats *ps, bool isEnter) {
  //clog << timeStamp() << "INFO  PatternA::setAvatars with grp " << PeopleStats::curAvatarGroup << "\n";
  ps->addAvatar(new AV9Square(ps, ps->inpterps), PeopleStats::curAvatarGroup);
}