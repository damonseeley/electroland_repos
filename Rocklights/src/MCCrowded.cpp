#include "MCCrowded.h"


MCCrowded::MCCrowded(PeopleStats *peopleStats) : MasterController(peopleStats) {

timeToIntermission = CProfile::theProfile->Int("intermissionSpacing", 5000);

  populatedThresh = CProfile::theProfile->Int("crowdedToPopulated", 14);


   //    curAmbient = new Ambient();

//     cou t << "Pattern1Square " << curAvMode << endl;
//  peopleStats->transitionTo(new Pattern1Square(1, 500, 1), 2000);
  peopleStats->transitionTo(new PatternA(), 2000);

  curAmbient = new AmbientA(true, true);
//  curAmbient = new AmbientA(true);
//  peopleStats->transitionTo(new PatternA(true), 2000);

}

void MCCrowded::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  timeToIntermission -= dt;
 if(! isInTransition) {
   if (worldStats->peopleCnt <= populatedThresh) {
      new MCPopulated(ps);
   } else if (timeToIntermission <= 0) {
      new MCTargetFlash(ps);
    }
  }
}



