/*
 *  Interpolators.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#ifndef __INTERPOLATORS_H__
#define __INTERPOLATORS_H__

#include <vector>

#include "Interpolator.h"

class Interpolator;

class Interpolators {

	vector<Interpolator *> interps1;
	vector<Interpolator *> interps2;
	vector<Interpolator *> *curInterps;
	vector<Interpolator *> *oldInterps;

public:
  Interpolators();
  ~Interpolators();
  void add(Interpolator *interp);
  void remove(Interpolator *interp);
  void update(int curTime, int deltaTime, float scale = 1.0f);
//  static Interpolators *theInterpolators;
}

;
#endif