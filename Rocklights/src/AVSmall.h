/*
 *  AVPerson.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */
#ifndef __AVSMALL_H__
#define __AVSMALL_H__

#include "Avatar.h"
#include "PersonStats.h"
#include "Panels.h"
#include "Interpolators.h"
#include "IGeneric.h"
#include "BasePixel.h"

class Avatar; 

class AVSmall : public Avatar  {
  
public:
  AVSmall(PersonStats *personStats,  Interpolators *interps);
  void updateFrame(PersonStats *personStats,  int ct, int dt, Interpolators *interps);
}
;
#endif