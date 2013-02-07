/*
 *  LETargetCircle.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */


#ifndef __LETARGETCIRCLE_H__
#define __LETARGETCIRCLE_H__

#include "LightElement.h"
#include <iostream>
#include <stdlib.h>
using namespace std;

class LETargetCircle : public LightElement {
  
public:
  unsigned char *r;
  LETargetCircle() { lightType = TARGET;}
  LETargetCircle(unsigned char *data, int channelOffset);
  //LETargetCircle(unsigned char *pixel);


  void setColor(unsigned char rVal, unsigned char gVal, unsigned char bVal);
  void setColor(unsigned char rVal);
  
  void getRGBData(unsigned char &cr, unsigned char &cg, unsigned char &cb);
  void setDataChannel(unsigned char *data, int channelOffset);

}
;

#endif
