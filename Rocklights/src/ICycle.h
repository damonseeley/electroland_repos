/*
 *  ICycle.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/29/05.
 * 
 *
 */


#ifndef __ICYCLE_H__
#define __ICYCLE_H__

#include "Interpolator.h"
#include "Interpolators.h"

#include <iostream>
#include <stdlib.h>
using namespace std;


class ICycle : public Interpolator {
  BasePixel *pixel;

  unsigned char rs;
  unsigned char gs;
  unsigned char bs;
  
  unsigned char re;
  unsigned char ge;
  unsigned char be;
  
  int holdTimeS;
  int holdTimeE;
  
  int transTimeStoE;
  int transTimeEtoS;
  
  float curR, curG, curB;

  float sToEPerMilr;
  float sToEPerMilg;
  float sToEPerMilb;

  float eToSPerMilr;
  float eToSPerMilg;
  float eToSPerMilb;
  
  int cycle;
  int curTime;
  int mode;
  int firstMode;
  
  enum { START, STOE, END, ETOS };
  
  
public:
    
    ICycle(Interpolators *q, BasePixel *pix, 
                 unsigned char rStart, unsigned char gStart, unsigned char bStart, int startUSecs, int startToEndUSecs,
                 unsigned char rEnd, unsigned char gEnd, unsigned char bEnd, int endUSecs, int endToStartUSecs, int cycleCnt);
  
  void update(int curTime, int deltaTime);
  void enterStart();
  void enterStoe();
  void enterEnd();
  void enterEtos();
}
;

#endif