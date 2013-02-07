#include "MCEmpty.h"

MCEmpty::MCEmpty(PeopleStats *peopleStats) : MasterController(peopleStats) {

  timeToIntermission = CProfile::theProfile->Int("intermissionSpacing", 5000);
  transThresh = CProfile::theProfile->Int("emptyToPopulated", 1);

  curAmbient = AmbientCreatorHash::theACHash->create("AmbRedStick");

  peopleStats->transitionTo(new Pattern(), 2000);
}

void MCEmpty::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  timeToIntermission -= dt;
  if(! isInTransition) {
    if(worldStats->peopleCnt >= transThresh) {
      new MCPopulated(ps);
    } else if (timeToIntermission <= 0) {
      new MCTargetFlash(ps);
    }
  }
}
