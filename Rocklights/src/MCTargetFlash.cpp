#include "MCTargetFlash.h"

MCTargetFlash::MCTargetFlash(PeopleStats *peopleStats) : MasterController(peopleStats) {

//  timeToIntermission = CProfile::theProfile->Int("intermissionSpacing", 5000);
  populatedThresh = CProfile::theProfile->Int("emptyToPopulated", 1);
  crowdedThresh = CProfile::theProfile->Int("populatedToCrowded", 20);

  curAmbient = new AmbTargetFlash(true);

  peopleStats->transitionTo(new Pattern(), 2000);
  timeToIntermission = CProfile::theProfile->Int("targetFlashIntTime", 10000);;
}

void MCTargetFlash::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  timeToIntermission -= dt;
  if (timeToIntermission <= 0) {
    if(worldStats->peopleCnt >= crowdedThresh) {
      new MCCrowded(ps);
    } else if (worldStats->peopleCnt >= populatedThresh) {
      new MCPopulated(ps);
    } else {
      new MCEmpty(ps);
    }
  }
}

