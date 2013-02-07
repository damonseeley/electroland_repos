#include "MCSinglePixelDance.h"

MCSinglePixelDance::MCSinglePixelDance(PeopleStats *peopleStats) : MasterController(peopleStats) {
  curAmbient = new Ambient();
  peopleStats->transitionTo(new PatternPixelDance(), 2000);
}

void MCSinglePixelDance::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
 
}
