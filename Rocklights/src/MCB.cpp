#include "MCB.h"

MCB::MCB(PeopleStats *peopleStats) : MasterController(peopleStats) {
  curAmbient = NULL;

  peopleStats->transitionTo(new PatternB(), 2000);
}

void MCB::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
  if (worldStats->peopleCnt > 4) {
    new MCA(ps);
  }
}
