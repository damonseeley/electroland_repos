#include "PatternPlusSign.h"

void PatternPlusSign::setAvatars(PersonStats *ps, bool isEnter) {
//  co ut << "PatternA::setAvatars with grp " << PeopleStats::curAvatarGroup << endl;
  ps->addAvatar(new AVPlusSign(ps, ps->inpterps), PeopleStats::curAvatarGroup);
}