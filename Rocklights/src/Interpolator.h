/*
 *  Interpolator.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */
#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

#include <iostream>
#include "BasePixel.h"
#include "Avatar.h"

#include "OffsetPixel.h"
#include "AmbientPixel.h"
#include "Interpolators.h"


class Interpolators;
class OffsetPixel;
class AmbientPixel;
class Avatar;

class Interpolator {
public:
		bool needsReaping;

	bool isGeneric;
  Interpolators *queue;
  Interpolator *next;
  Interpolator *prev;


  Avatar *interpDoneListener;

  Interpolator();

  virtual ~Interpolator();
  virtual void update(int curTime, int deltaTime, float scale) {}  // return true if delete
  virtual void notifyPixelDeletion(OffsetPixel *op) {}
  virtual void notifyPixelDeletion(AmbientPixel *op) {}
  void setDoneListener(Avatar *a) { interpDoneListener = a; }
}
;
#endif