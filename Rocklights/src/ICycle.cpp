/*
 *  ICycle.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/29/05.
 * 
 *
 */

#include "ICycle.h"

ICycle::ICycle(Interpolators *q, BasePixel *pix, 
               unsigned char rStart, unsigned char gStart, unsigned char bStart, int startUSecs, int startToEndUSecs,
               unsigned char rEnd, unsigned char gEnd, unsigned char bEnd, int endUSecs, int endToStartUSecs, int cycleCnt) {
  if(pix != NULL) {
  queue = q;
  pixel = pix;
  
  rs = rStart;
  gs = gStart;
  bs = bStart;
  
  re = rEnd;
  ge = gEnd;
  be = bEnd;
  
  holdTimeS = startUSecs;
  holdTimeE = endUSecs;
  
  transTimeStoE =startToEndUSecs;
  transTimeEtoS = endToStartUSecs;
  
  cycle = cycleCnt;
  
  
  if(transTimeStoE > 0) {
    float scale = 1.0f / (float) transTimeStoE;
    sToEPerMilr = ((float) (re - rs)) * scale;
    sToEPerMilg = ((float) (ge - gs)) * scale;
    sToEPerMilb = ((float) (be - bs)) * scale;
  } else {
    sToEPerMilr = 0;
    sToEPerMilg = 0;
    sToEPerMilb = 0;
  }
  
  if(transTimeEtoS > 0) {
    float scale = 1.0f / (float) transTimeEtoS;
    eToSPerMilr = ((float) (rs - re)) * scale;
    eToSPerMilg = ((float) (gs - ge)) * scale;
    eToSPerMilb = ((float) (bs - be)) * scale;
  } else {
    eToSPerMilr = 0;
    eToSPerMilg = 0;
    eToSPerMilb = 0;
  }
  
    if (holdTimeS > 0) {
      firstMode = START;
      enterStart();
    } else if (transTimeStoE > 0) {
      firstMode = STOE;
      enterStoe();
    } else if (holdTimeE > 0) {
      firstMode = END;
      enterEnd();
    } else if (holdTimeE > 0) {
      firstMode = ETOS;
      enterEtos();
    } 
    
  
  enterStart();
  queue->add(this);
  } else {
    delete this;
  }
}

void ICycle::enterStart() {
  mode = START;
  if(mode == firstMode) {
    cycle--;
  }
  curTime = holdTimeS;
  curR = rs;
  curG = gs;
  curB = bs;
}

void ICycle::enterStoe() {
  mode = STOE;
  if(mode == firstMode) {
    cycle--;
  }
  curTime = transTimeStoE;
  curR = rs;
  curG = gs;
  curB = bs;
}

void ICycle::enterEnd() {
  mode = END;
  if(mode == firstMode) {
    cycle--;
  }
  curTime = holdTimeE;
  curR = re;
  curG = ge;
  curB = be;  
}

void ICycle::enterEtos() {
  mode = ETOS;
  if(mode == firstMode) {
    cycle--;
  }
  curTime = transTimeEtoS;
  curR = re;
  curG = ge;
  curB = be;
}


void ICycle::update(int sysTime, int deltaTime) {
  if (cycle < 0) {
    delete this;
    return;
  } else {
    pixel->addColor((unsigned char)curR, (unsigned char)curG, (unsigned char)curB);      
    switch(mode) {
      case START:
        if (curTime > 0) {
          curTime -= deltaTime;
        } else {
          if (transTimeStoE > 0) {
            enterStoe();
          } else if (holdTimeE > 0) {
            enterEnd();
          } else if (transTimeEtoS > 0) {
            enterEtos();
          } else { // stay in start and just cycle
            enterStart();
          }
        }
        break;
      case STOE:
        if(curTime > 0) {
          curTime -= deltaTime;
          curR += sToEPerMilr * deltaTime;
          curG += sToEPerMilg * deltaTime;
          curB += sToEPerMilb * deltaTime;
          
          
          curR = (curR > 255) ? 255 : curR;
          curG = (curG > 255) ? 255 : curG;
          curB = (curB > 255) ? 255 : curB;
          curR = (curR < 0) ? 0 : curR;
          curG = (curG < 0) ? 0 : curG;
          curB = (curB < 0) ? 0 : curB;
          
        } else {
          if (holdTimeE > 0) {
            enterEnd();
          } else if (transTimeEtoS > 0) {
            enterEtos();
          } else if (holdTimeS > 0) {
            enterStart();
          } else {
            enterStoe();
          }
        }
        break;
      case END:
        if (curTime > 0) {
          curTime -= deltaTime;
        } else {
          if (transTimeEtoS > 0) {
            enterEtos();
          } else if (holdTimeS > 0) { // stay in start and just cycle
            enterStart();
          } else if (transTimeStoE > 0) {
            enterStoe();
          } else {
            enterEnd();
          } 
        }
        break;
      case ETOS:
        if (curTime > 0) {
          curTime -= deltaTime;
          curR += eToSPerMilr * deltaTime;
          curG += eToSPerMilg * deltaTime;
          curB += eToSPerMilb * deltaTime;

          curR = (curR > 255) ? 255 : curR;
          curG = (curG > 255) ? 255 : curG;
          curB = (curB > 255) ? 255 : curB;
          curR = (curR < 0) ? 0 : curR;
          curG = (curG < 0) ? 0 : curG;
          curB = (curB < 0) ? 0 : curB;
          
        } else {
          if (holdTimeS > 0) {
            enterStart();
          } else if (transTimeStoE > 0) {
            enterStoe();
          } else if (holdTimeE > 0) {
            enterEnd();
          } else {
            enterEtos();
          } 
        }
        break;      
    }    
  }
  
  
}