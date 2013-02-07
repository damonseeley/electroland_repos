#include "MCPopulated.h"
int MCPopulated::curAvMode = -1;

MCPopulated::MCPopulated(PeopleStats *peopleStats) : MasterController(peopleStats) {

  avatarCycleTime =   CProfile::theProfile->Int("avatarCycleTime", 0);

   emptyThresh = CProfile::theProfile->Int("populatedToEmpty", 0);
   crowdedThresh = CProfile::theProfile->Int("populatedToCrowded", 15);
   
  curAvMode++;

  curAvMode =  (curAvMode > 4) ? 0 : curAvMode;

  timeLeft = avatarCycleTime;

     curAmbient = new Ambient();

//     cou t << "Pattern1Square " << curAvMode << endl;
  peopleStats->transitionTo(new Pattern1Square(curAvMode, 500, 1), 2000);

}

void MCPopulated::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
 if(! isInTransition) {
    if(worldStats->peopleCnt >= crowdedThresh) {
      new MCCrowded(ps);
    } else if (worldStats->peopleCnt <= emptyThresh) {
      new MCEmpty(ps);
    } 
    // UNDONE REMOVE THIS CODE FOR TRANSITONS TO OTHER POPULATED
    //else if (timeLeft <= 0) {
    //  new MCPopulated(ps);
    //}
  }
// timeLeft -= dt; 
}
