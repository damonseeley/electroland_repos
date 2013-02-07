/*
 *  LightElement.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#ifndef __LIGHTELEMENT_H__
#define __LIGHTELEMENT_H__

class LightElement {
  
public:
  
  enum LightTypes { BLANK, RGB, TARGET };
  int lightType;

  LightElement() { lightType = BLANK;}
  virtual ~LightElement() {}
  virtual void setColor(unsigned char rVal, unsigned char gVal, unsigned char bVal) {}
  virtual void clear() {}
  virtual void getRGBData(unsigned char &cr, unsigned char &cg, unsigned char &cb){}
  
  virtual void setDataChannel(unsigned char *data, int channelOffset) {}

  inline int getMax(int a, int b, int c) { return (a > b) ? ((a > c) ? a : c)  : ((b > c) ? b : c); }

}
;

#endif