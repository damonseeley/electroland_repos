/*
 *  IHoldAndFade.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#ifndef __IHOLDANDFADE_H__
#define __IHOLDANDFADE_H__

#include "Interpolator.h"
#include "Interpolators.h"

#include <iostream>
#include <stdlib.h>
using namespace std;


class IHoldAndFade : public Interpolator {
  BasePixel *pixel;
  
  unsigned char re;
  unsigned char ge;
  unsigned char be;
  
  int holdTime;
  
  float curR, curG, curB;
  float fadePerMilr;
  float fadePerMilg;
  float fadePerMilb;

  int fadeTime;
  
  
public:
  
  IHoldAndFade(Interpolators *q, BasePixel *pix, unsigned char rStart, unsigned char gStart, unsigned char bStart, int holdUSecs, unsigned char rEnd, unsigned char gEnd, unsigned char bEnd, int fadeUSecs);
  void update(int curTime, int deltaTime);
}
;

#endif