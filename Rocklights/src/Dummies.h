// Dummies.h: interface for the Dummies class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __DUMMIES_H__
#define __DUMMIES_H__

#include "PersonStats.h"
#include "PeopleStats.h"
#include "WorldStats.h"

//#include "Pattern.h"
//#include "PatternA.h"
//#include "PatternB.h"
#include "MCA.h"
#include "MCB.h"


class Dummies {

  int roomMaxX;
  int roomMaxY;

  struct Dummy {
    int personId;
    float x;
    float y;
    float z;

    float dx;
    float dy;


  };

  PeopleStats *people;
  WorldStats *world;
  Dummy dummy[DUMMYCNT];
  
  Dummy *selected;

  int lastDummy;

  enum { PATA, PATB };
//  int tmpPat;


public:
  Dummies(PeopleStats *p, WorldStats *w) ;
	~Dummies();

  bool setSelectMode(int i) ;
  void update(float x, float y);

  bool inBounds(float x, float y);

  void setRoomDim(int x, int y) { roomMaxX = x; roomMaxY = y; }

  void genDummys(int cnt);
  void updateDummys();



};

#endif // __DUMMIES_H__
