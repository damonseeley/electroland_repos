#ifndef __MCPOPULATED_H__
#define __MCPOPULATED_H__

#include "MasterController.h"

#include "MCSparce.h"
#include "MCCrowded.h"
#include "PatternPlusSign.h"

class MCPopulated : public MasterController {
  int crowdedThresh;
  int emptyThresh;
  int avatarCycleTime;
  static int curAvMode ;
  int timeLeft;

public:
  MCPopulated(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
