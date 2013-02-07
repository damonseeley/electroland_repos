/*
 *  ColorElementR.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/25/05.
 * 
 *
 */

#ifndef _COLORELEMENTR_H_
#define _COLORELEMENTR_H_


#include "ColorElementRGB.h"
#include<fstream>

class  ColorElementR : public ColorElementRGB {
  ColorElementR() ;
  virtual void update();
  virtual void render() {}
}
;
#endif
