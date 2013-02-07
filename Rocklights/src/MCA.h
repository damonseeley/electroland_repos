#ifndef __MCA_H__
#define __MCA_H__

#include "MasterController.h"
#include "AmbientA.h"
#include "PatternA.h"
#include "MCB.h"
#include "AV1Square.h"

class MCA : public MasterController {

public:
  MCA(PeopleStats *peopleStats);
  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

}
;
#endif
