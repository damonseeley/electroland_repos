#ifndef __AV4SQUARE_H
#define __AV4SQUARE_H

#include "Avatar.h"
#include "PersonStats.h"
#include "Interpolators.h"
#include "IGeneric.h"

class Avatar; 

class AV4Square : public Avatar  {
  static int red[];
  static int redAnim[];
  static int redAnim3Dots[];
public:
  AV4Square(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);
}
;
#endif