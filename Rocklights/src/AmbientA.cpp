#include "AmbientA.h"


int AmbientA::seas[] = { 
  0, 50, 0,  0, 45, 0,  100,     //1
  0, 45, 0,  0, 40, 5,  100,//2
  0, 40, 5,  0, 35, 10,  100,//3
  0, 35, 10,  0,30, 15,  100,//4
  0, 30, 15,  0, 25, 20,  100,//5
  0, 25, 20,  0, 20, 25,  100,//6
  0, 20, 25,  0, 15, 30,  100,//7
  0, 15, 30,  0, 10, 35,  100,//8
  0, 10, 35,  0, 5, 40,  100,//9
  0, 5, 40,  0, 0, 45,  100,//10
  0, 0, 45,  0, 0, 50,  100,//11
  0, 0, 50,  0, 5, 45,  100,//12
  0, 5, 45,  0, 10, 40,  100,//13
  0, 10, 40,  0, 15, 35,  100,//14
  0, 15, 35,  0, 20, 30,  100,//15
  0, 20, 30,  0, 25, 25,  100,//16
  0, 25, 25,  0, 30, 20,  100,//17
  0, 30, 20,  0, 35, 15,  100,//18
  0, 35, 15,  0, 40, 10,  100,//19
  0, 40, 10,  0, 45, 5,  100,//20
  0, 45, 5,  0, 50, 0,  100,//21
    -1
 
};

int AmbientA::seasR[] = { 
  50, 0, 0,   45,0,  0,  100,     //1
  45, 0, 0,   40,0,  5,  100,//2
 40, 0, 5,    35,0,  10,  100,//3
 35, 0, 10,   30,0,  15,  100,//4
 30, 0, 15,   25,0,  20,  100,//5
 25, 0, 20,   20,0,  25,  100,//6
 20, 0, 25,   15,0,  30,  100,//7
 15, 0, 30,   10,0,  35,  100,//8
 10, 0, 35,   5, 0, 40,  100,//9
 5, 0, 40,    0, 0, 45,  100,//10
 0, 0, 45,    0, 0, 50,  100,//11
 0, 0, 50,    5, 0, 45,  100,//12
 5, 0, 45,    10, 0, 40,  100,//13
 10, 0, 40,   15, 0, 35,  100,//14
 15, 0, 35,   20, 0, 30,  100,//15
 20, 0, 30,   25, 0, 25,  100,//16
 25, 0, 25,   30, 0, 20,  100,//17
 30, 0, 20,   35, 0, 15,  100,//18
 35, 0, 15,   40, 0, 10,  100,//19
 40, 0, 10,   45, 0, 5,  100,//20
 45, 0, 5,    50, 0, 0,  100,//21
    -1
 
};



bool AmbientA::inited = false;

AmbientA::AmbientA(bool create, bool green) : Ambient(create) {
  if(! inited) {
  for(int i = 0; i < 21; i++) {
    int row = i * 7;
    seas[row + 1] *= 1.5;
    seas[row + 2] *= 1.5;
    seas[row + 4] *= 1.5;
    seas[row + 5] *= 1.5;
    seasR[row + 1] *= 1.5;
    seasR[row + 2] *= 1.5;
    seasR[row + 4] *= 1.5;
    seasR[row + 5] *= 1.5;
  }
  inited = true;
  }

  
 
  for(int p = 0; p < Panels::PANEL_CNT; p++) {
  for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
    for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
      for(int s = 0; s < 4; s++) { 
        if(! Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
          new IGeneric(interps, addAmbientPixel(p, c, r, s), seas, -1, random(20));
        }
      }
    }
  }
}
}