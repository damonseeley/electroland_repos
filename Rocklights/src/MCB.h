#ifndef __MCB_H__
#define __MCB_H__

#include "MasterController.h"
#include "PatternB.h"
#include "MCA.h"

class MCB : public MasterController {

public:
  MCB(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
