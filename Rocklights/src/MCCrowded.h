#ifndef __MCCROWDED_H__
#define __MCCROWDED_H__

#include "MasterController.h"
#include "MCMobbed.h"
#include "MCPopulated.h"
#include "Pattern1Square.h"

class MCCrowded : public MasterController {
    int timeToIntermission;

//  static bool isGreen;
  int populatedThresh;

public:
  MCCrowded(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
