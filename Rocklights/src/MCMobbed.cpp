#include "MCMobbed.h"

MCMobbed::MCMobbed(PeopleStats *peopleStats) : MasterController(peopleStats) {
//  curAmbient = NULL;
//  peopleStats->transitionTo(new PatternB(), 2000);
  populatedThresh = CProfile::theProfile->Int("mobbedToCrowded", 25);
   
  
  curAmbient = new AmbientA(true, true);
  peopleStats->transitionTo(new PatternA(), 2000);

}

void MCMobbed::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  if(! isInTransition) {
    if (worldStats->peopleCnt <= populatedThresh) {
      new MCCrowded(ps);
    }
  }
}
