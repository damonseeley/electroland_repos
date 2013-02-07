#ifndef __MCTARGETFLASH_H__
#define __MCTARGETFLASH_H__

#include "MasterController.h"
#include "MCSparce.h"
#include "MCEmpty.h"
#include "AmbTargetFlash.h"
#include "Pattern.h"


class MCTargetFlash : public MasterController {
  int populatedThresh ;
  int crowdedThresh ;

  int timeToIntermission;
public:
  MCTargetFlash(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
