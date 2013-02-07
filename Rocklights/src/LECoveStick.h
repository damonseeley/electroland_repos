/*
 *  LECoveStick.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#ifndef __LECOVESTICK_H__
#define __LECOVESTICK_H__

#include "LightElement.h"

class LECoveStick : public LightElement {
public:
  unsigned char *r;
  unsigned char *g;
  unsigned char *b;

  LECoveStick( ) { lightType = RGB; }
  LECoveStick(unsigned char *data, int channelOffset);
 
  
  
  
  void setColor(unsigned char rVal, unsigned char gVal, unsigned char bVal);
  void getRGBData(unsigned char &cr, unsigned char &cg, unsigned char &cb);
  
  void setDataChannel(unsigned char *data, int channelOffset);

}
;

#endif
