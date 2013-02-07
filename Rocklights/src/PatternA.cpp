#include "PatternA.h"

PatternA::PatternA() {
}

void PatternA::setAvatars(PersonStats *ps, bool isEnter) {
    Avatar *a = new AVHuge(ps, ps->inpterps);
  
  //UNDONE CHANGE TO VARS
  a->setPillarMode(1);
//  a->setTrailDelay(tailDelay);
  
  ps->addAvatar(a, PeopleStats::curAvatarGroup);

//  co ut << "PatternA::setAvatars with grp " << PeopleStats::curAvatarGroup << endl;
//  ps->addAvatar(new AVHuge(ps, ps->inpterps), PeopleStats::curAvatarGroup);
}