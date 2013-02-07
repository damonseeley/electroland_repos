/*
 *  IHoldAndFade.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#include "IHoldAndFade.h"




IHoldAndFade::IHoldAndFade(Interpolators *q, BasePixel *pix, unsigned char rStart, unsigned char gStart, unsigned char bStart, int holdUSecs, unsigned char rEnd, unsigned char gEnd, unsigned char bEnd, int fadeUSecs) {
  if(pix != NULL) {
    queue = q;
    pixel = pix;
    
    curR = rStart;
    curG = gStart;
    curB = bStart;
    
    re = rEnd ;
    ge = gEnd ;
    be = bEnd ;
    
    
    holdTime = holdUSecs;
    fadeTime = fadeUSecs;
    
    if(fadeUSecs != 0) {
      float scale = 1.0f / (float) fadeUSecs;
      fadePerMilr = ((float) (re - rStart)) * scale;
      fadePerMilg =((float)(ge - gStart)) * scale;
      fadePerMilb =((float)(be - bStart)) * scale;
    }
    queue->add(this);
  } else {
    delete this;
  }
  
}


void IHoldAndFade::update(int curTime, int deltaTime) {
  if(holdTime >= 0) {
    pixel->addColor((unsigned char)curR, (unsigned char)curG, (unsigned char)curB);      
    holdTime -= deltaTime;
  } else if (fadeTime >= 0) {
    pixel->addColor((unsigned char)curR, (unsigned char)curG, (unsigned char)curB);      
    fadeTime -= deltaTime;
    curR += (fadePerMilr * deltaTime);
    curG += (fadePerMilg * deltaTime);
    curB += (fadePerMilb * deltaTime);
  } else {
    pixel->addColor(re, ge, be); 
    delete this;
    return;
  }
}



