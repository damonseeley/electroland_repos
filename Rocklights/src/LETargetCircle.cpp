/*
 *  LETargetCircle.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#include "LETargetCircle.h"

LETargetCircle::LETargetCircle(unsigned char *data, int offSet) {
  r = &data[offSet];  
  lightType = TARGET;
}


void LETargetCircle::setDataChannel(unsigned char *data, int offSet) {
  r = &data[offSet];  
}


void LETargetCircle::setColor(unsigned char rVal, unsigned char gVal, unsigned char bVal) {
  int rt = (rVal > gVal) ? rVal : gVal;
  rt = (rt > bVal) ? rt : bVal;
  *r = rt;
 // cou t << "r set to " << (int) *r << "   b:" << (int) bVal << endl;
}

void LETargetCircle::setColor(unsigned char rVal) {
  *r = rVal;
}


void LETargetCircle::getRGBData(unsigned char &cr, unsigned char &cg, unsigned char &cb) {
  cr = *r;
  cg = 0;
  cb = 0;
}


