#ifndef __AV1SQUAREKIT_H
#define __AV1SQUAREKIT_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"

class Avatar; 

class AV1SquareKit : public Avatar  {

static int kit[];


public:
  AV1SquareKit(PersonStats *personStats,  Interpolators *interps);
}
;
#endif
