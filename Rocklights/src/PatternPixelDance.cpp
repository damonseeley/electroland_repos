#include "PatternPixelDance.h"

void PatternPixelDance::setAvatars(PersonStats *ps, bool isEnter) {
//  co ut << "PatternA::setAvatars with grp " << PeopleStats::curAvatarGroup << endl;
  ps->addAvatar(new AVSinglePixelDance(ps, ps->inpterps), PeopleStats::curAvatarGroup);
}