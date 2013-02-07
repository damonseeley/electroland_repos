#include "PatternB.h"

void PatternB::setAvatars(PersonStats *ps, bool isEnter) {
//  co ut << "PatternB::setAvatars with grp " << PeopleStats::curAvatarGroup << endl;
 ps->addAvatar(new AV9Square(ps, ps->inpterps), PeopleStats::curAvatarGroup);
//  ps->addAvatar(new AVHuge(ps, ps->inpterps), PeopleStats::curAvatarGroup);
}