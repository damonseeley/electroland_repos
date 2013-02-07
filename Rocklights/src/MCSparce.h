#ifndef __MCSPARCE_H__
#define __MCSPARCE_H__

#include "MasterController.h"
#include "MCPopulated.h"
#include "MCEmpty.h"
#include "Pattern9Square.h"

class MCSparce : public MasterController {

  int emptyThresh;
  int populatedThresh;

public:

  MCSparce(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
