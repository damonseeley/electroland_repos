#ifndef __MCSINGLEPIXELDANCE_H__
#define __MCSINGLEPIXELDANCE_H__

#include "MasterController.h"
#include "PatternPixelDance.h"

class MCSinglePixelDance : public MasterController {

public:
  MCSinglePixelDance(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
