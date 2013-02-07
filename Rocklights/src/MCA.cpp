#include "MCA.h"

MCA::MCA(PeopleStats *peopleStats) : MasterController(peopleStats) {
  curAmbient = new AmbientA(true, true);

  peopleStats->transitionTo(new PatternA(), 2000);

}


void MCA::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
//  if (worldStats->peopleCnt < 3) {
//    new MCB(ps);
//  }
}
