/*
 *  IGeneric.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __IGENERIC_H__
#define __IGENERIC_H__

#include "Interpolator.h"
#include "Avatar.h"
#include "Interpolators.h"

#include <iostream>
#include <stdlib.h>
using namespace std;



class IGeneric : public Interpolator {
  bool advance();

  void setup(Interpolators *q, BasePixel *pix, int *path, int loop, float startFrame);
public:
    BasePixel *pixel;
  int *l;
  int curI;
  
  int loop;

  enum { R1, G1, B1, R2, G2, B2, T } ;
  
  float curR, curG, curB;
  float dR, dG, dB;
  
  int r2;
  int g2;
  int b2;
  int t;

  static int tmpNextID;
  int id;

  float timeScale; // 1.0 regular sepeed, .5 half as fast, 2.0 twice as fast

//  Avatar *doneListener;

  
  // path is and arbirarly long list of r g b r2 g2 b2 times
  // transitions from rgb to r2g2b2 in time milisecs until done
  // end with a -1
//  IGeneric(Interpolators *q, BasePixel *pix, int *path);
  IGeneric(Interpolators *q, BasePixel *pix, int *path, int loop = 1);
  IGeneric(Interpolators *q, BasePixel *pix, int *path, int loop, float startFrame);

  IGeneric(Interpolators *q, OffsetPixel *pix, int *path, int loop = 1);
  IGeneric(Interpolators *q, OffsetPixel *pix, int *path, int loop, float startFrame);



    IGeneric(Interpolators *q, AmbientPixel *pix, int *path, int loop = 1);
  IGeneric(Interpolators *q, AmbientPixel *pix, int *path, int loop, float startFrame);

   ~IGeneric();

  void update(int curTime, int deltaTime, float scale = 1.0f);
  bool setLoop(int i) { loop = i; }
  virtual void notifyPixelDeletion(OffsetPixel *op);
  virtual void notifyPixelDeletion(AmbientPixel *op);

 // void addDoneListener(Avatar *a);
}
;
#endif