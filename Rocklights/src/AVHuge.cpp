#include "AVHuge.h"

int AVHuge::seas[] = { 
  0, 50, 0,  0, 45, 0,  75,     //1
  0, 45, 0,  0, 40, 5,  50,//2
  0, 40, 5,  0, 35, 10,  100,//3
  0, 35, 10,  0,30, 15,  79,//4
  0, 30, 15,  0, 25, 20,  100,//5
  0, 25, 20,  0, 20, 25,  110,//6
  0, 20, 25,  0, 15, 30,  100,//7
  0, 15, 30,  0, 10, 35,  89,//8
  0, 10, 35,  0, 5, 40,  100,//9
  0, 5, 40,  0, 0, 45,  100,//10
  0, 0, 45,  0, 0, 50,  76,//11
  0, 0, 50,  0, 5, 45,  100,//12
  0, 5, 45,  0, 10, 40,  100,//13
  0, 10, 40,  0, 15, 35,  89,//14
  0, 15, 35,  0, 20, 30,  100,//15
  0, 20, 30,  0, 25, 25,  87,//16
  0, 25, 25,  0, 30, 20,  100,//17
  0, 30, 20,  0, 35, 15,  96,//18
  0, 35, 15,  0, 40, 10,  100,//19
  0, 40, 10,  0, 45, 5,  79,//20
  0, 45, 5,  0, 50, 0,  93,//21
    -1
 
};

int AVHuge::seasLight[] = { 
  0, 50, 0,  0, 45, 0,  75,     //1
  0, 45, 0,  0, 40, 5,  50,//2
  0, 40, 5,  0, 35, 10,  100,//3
  0, 35, 10,  0,30, 15,  79,//4
  0, 30, 15,  0, 25, 20,  100,//5
  0, 25, 20,  0, 20, 25,  110,//6
  0, 20, 25,  0, 15, 30,  100,//7
  0, 15, 30,  0, 10, 35,  89,//8
  0, 10, 35,  0, 5, 40,  100,//9
  0, 5, 40,  0, 0, 45,  100,//10
  0, 0, 45,  0, 0, 50,  76,//11
  0, 0, 50,  0, 5, 45,  100,//12
  0, 5, 45,  0, 10, 40,  100,//13
  0, 10, 40,  0, 15, 35,  89,//14
  0, 15, 35,  0, 20, 30,  100,//15
  0, 20, 30,  0, 25, 25,  87,//16
  0, 25, 25,  0, 30, 20,  100,//17
  0, 30, 20,  0, 35, 15,  96,//18
  0, 35, 15,  0, 40, 10,  100,//19
  0, 40, 10,  0, 45, 5,  79,//20
  0, 45, 5,  0, 50, 0,  93,//21
    -1
 
};
/*

int AVHuge::seasR[] = { 
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
*/
bool AVHuge::init = false;

AVHuge::AVHuge(PersonStats *personStats, Interpolators *interps) : Avatar () {

  if(init == false) {
 for(int i = 0; i < 21; i++) {
    int row = i * 7;
    seas[row + 1] *= 3;
    seas[row + 2] *= 3;
    seas[row + 4] *= 3;
    seas[row + 5] *= 3;
    seasLight[row + 1] *= 2;
    seasLight[row + 2] *= 2;
    seasLight[row + 4] *= 2;
    seasLight[row + 5] *= 2;
  }
 init = true;
  }
  if(random(2) >= 1) {
  setColor(Avatar::BLUE);
  } else {
  setColor(Avatar::GREEN);
  }

  new IGeneric(interps, addOffsetPixel(A, 0, 0), seas, -1, random(20));

  if (random(100) > 50) {
    new IGeneric(interps, addOffsetPixel(A, 1, 0), seasLight, -1, random(20));
  }
  if (random(100) > 50) {
  new IGeneric(interps, addOffsetPixel(A, 0, 1), seasLight, -1, random(20));
  }
  if (random(100) > 50) {
  new IGeneric(interps, addOffsetPixel(A, -1, 0), seasLight, -1, random(20));
  }
  if (random(100) > 50) {
  new IGeneric(interps, addOffsetPixel(A, 0, -1), seasLight, -1, random(20));
  }
  if (random(100) > 50) {  
  new IGeneric(interps, addOffsetPixel(A, 1, 1), seasLight, -1, random(20));
  }
  if (random(100) > 50) {  
  new IGeneric(interps, addOffsetPixel(A, -1, -1), seasLight, -1, random(20));
  }
  if (random(100) > 50) {  
  new IGeneric(interps, addOffsetPixel(A, 1, -1), seasLight, -1, random(20));
  }
    if (random(100) > 50) {  
  new IGeneric(interps, addOffsetPixel(A, -1, 1), seasLight, -1, random(20));
    }
    
 
}
 



void AVHuge::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
//  int col = personStats->col;
//  int row = personStats->row;

//  if(personStats->pixel->isTarget) {
//    personStats->pixel->addColor(255, 0, 0);
//  }

  
}