#ifndef __MCEMPTY_H__
#define __MCEMPTY_H__

#include "MasterController.h"
#include "MCSparce.h"
#include "MCTargetFlash.h"
#include "AmbientCreatorHash.h"

class MCEmpty : public MasterController {
  int transThresh;
  int timeToIntermission;
public:
  MCEmpty(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
