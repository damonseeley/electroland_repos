#ifndef __MCMOBBED_H__
#define __MCMOBBED_H__

#include "MasterController.h"
#include "MCPopulated.h"
#include "AmbientA.h"
#include "PatternA.h"

class MCMobbed : public MasterController {
  int populatedThresh;


public:
  MCMobbed(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
