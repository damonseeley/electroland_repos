/*
 *  ColorElement.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/24/05.
 * 
 *
 */

#ifndef _COLORELEMENT_H_
#define _COLORELEMENT_H_

class  ColorElement {
protected:
  unsigned char uR;
  unsigned char uG;
  unsigned char uB;
  
  int ipIndex;
  int channel;
  int addType;  
  
  
public:
  virtual ~ColorElement() {}
  enum addMethodTypes { CAP, MAX, NORM, AND, OR, XOR, OVERWRITE, AVERAGE };

  
  
  virtual void setColor(unsigned char cr, unsigned char cg, unsigned char cb) {}
  virtual void setR(unsigned char c) {}
  virtual void setG(unsigned char c) {}
  virtual void setB(unsigned char c) {}
  
  virtual void addColor(unsigned char cr, unsigned char cg, unsigned char cb) {}
  
  virtual void getColor(unsigned char &cr, unsigned char &cg, unsigned char &cb)  { cr = uR; cg = uG; cb = uB; }
  inline virtual unsigned char getR() { return uR;}
  inline virtual unsigned char getG() { return uG;}
  inline virtual unsigned char getB() { return uB;}

  virtual void clear() {}
  
  void setipIndex(int i) { ipIndex = i; }
  void setChannel(int i) { channel = i; }
  void setAddType(int i) { addType = i; }
  
  int getipIndex() { return ipIndex; }
  int getChannel() { return channel; }
  int getAddType() { return addType; }
  
  virtual void update() {} 
  virtual void render() {} 
}
;

#endif
