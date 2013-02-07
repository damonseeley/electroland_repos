#include "MCSparce.h"




MCSparce::MCSparce(PeopleStats *peopleStats) : MasterController(peopleStats) {

  emptyThresh = CProfile::theProfile->Int("sparceToEmpty", 0);
  populatedThresh  = CProfile::theProfile->Int("sparceToPopulated", 5);

  curAmbient = new Ambient();
  peopleStats->transitionTo(new Pattern9Square(), 2000);
}

void MCSparce::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  if(! isInTransition) {
    if(worldStats->peopleCnt >= populatedThresh) {
      new MCPopulated(ps);
    } else if (worldStats->peopleCnt <= emptyThresh) {
      new MCEmpty(ps);
    }
  }
}
