#include "AVSinglePixelDance.h"

int AVSinglePixelDance::red[] = {
  255,  0,    0,    0, 0, 0, 200, 
    0,  0,    0,    255, 0, 0, 200, 
    -1};
  
  int AVSinglePixelDance::redFade[] = {
    255,  0,    0,    0, 0, 0, 500, 
      -1
  };
  
  
  int AVSinglePixelDance::redIn0[] = {
    0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn1[] = {
    0, 0, 0, 0, 0, 0, 60,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn2[] = {
    0, 0, 0, 0, 0, 0, 120,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn3[] = {
    0, 0, 0, 0, 0, 0, 180,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn4[] = {
    0, 0, 0, 0, 0, 0, 240,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn5[] = {
    0, 0, 0, 0, 0, 0, 300,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  int AVSinglePixelDance::redIn6[] = {
    0, 0, 0, 0, 0, 0, 360,
      0, 0, 0, 255, 0, 0, 60,
      255, 0, 0, 255, 0, 0, INT_MAX,
      -1
  }
  ;
  
  
  AVSinglePixelDance::AVSinglePixelDance(PersonStats *personStats, Interpolators *interps) : Avatar () {
	  this->setColor(RED);
    new IGeneric(interps, addOffsetPixel(A, 0, 0, 0), red, -1, 0.0f);
    new IGeneric(interps, addOffsetPixel(A, 0, 0, 1), red, -1, .25f);
    new IGeneric(interps, addOffsetPixel(A, 0, 0, 2), red, -1, 0.5f);
    new IGeneric(interps, addOffsetPixel(A, 0, 0, 3), red, -1, .75f);
    //  new IGeneric(interps, addOffsetPixel(A, 0, -1), red, -1);
    c0 = NULL;
  }
  
  
  void AVSinglePixelDance::updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {
    /*
    
    int col = personStats->col;
    int row = personStats->row;
    
    // this is a trail (leave something to fade out if row and column change)
    if ((col != oldCol) || (row != oldRow)) {
      new IGeneric(interps, panels->getPixel(A, oldCol, oldRow), redFade, 1);
      oldCol = col;
      oldRow = row;
    }
    
    
    
    // this turns on pillars when one is close to them
    int pil = personStats->nearPillar;
      if (pil != oldPil) {
        if(c0) {
          delete c0;
          delete c1;
          delete c2;
          delete c3;
          delete c4;
          delete c5;
          delete c6;
          c0 = NULL;
        }
    
    if (pil != 0) {
      if (pil != oldPil) {

        int tPil = pil;
        if (tPil < 0) {
          row = -tPil;
          row -=1;
          tPil = 1;
        } else {
          row = 0;
        }
      
        
        c0 = new IGeneric(interps, panels->getPixel(tPil, 0, row), redIn6, 1);
        c1 = new IGeneric(interps, panels->getPixel(tPil, 1, row), redIn5, 1);
        c2 = new IGeneric(interps, panels->getPixel(tPil, 2, row), redIn4, 1);
        c3 = new IGeneric(interps, panels->getPixel(tPil, 3, row), redIn3, 1);
        c4 = new IGeneric(interps, panels->getPixel(tPil, 4, row), redIn2, 1);
        c5 = new IGeneric(interps, panels->getPixel(tPil, 5, row), redIn1, 1);
        c6 = new IGeneric(interps, panels->getPixel(tPil, 6, row), redIn0, 1);
      
        }
      }
    oldPil = pil;

    } 
  }
  /*
  row = 0;
  
    if (pil != 0) {
    if (pil < 0) {
    row = -pil;
    row -=1;
    pil = 1;
    }
    if (cRow = 0) {
    Panels::thePanels->getPixel(pil, col, row)->addColor(255, 0, 0);
    
      }
      
        /*  
        for (int col = 0; col < 7; col++) {
        Panels::thePanels->getPixel(pil, col, row)->addColor(255, 0, 0);
        }
  */
  
}