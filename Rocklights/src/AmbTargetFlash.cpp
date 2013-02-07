#include "AmbTargetFlash.h"
#include "MasterController.h"

int AmbTargetFlash::flashAndFade[] = {
  255, 0, 0, 0, 0, 0, 200,
  0,0,0,   0,0,0, INT_MAX,
    -1
}
;

AmbTargetFlash::AmbTargetFlash(bool ci) : Ambient(ci) {
string setting = MasterController::curMasterController->name+"TargetFlashHoldTime";
targetFlashHoldTime =  + CProfile::theProfile->Int(setting.c_str(), 400);
  /*
  for(int p = 0; p < Panels::PANEL_CNT; p++) {
  for (int c = 0; c < Panels::thePanels->panels[p].getWidth(); c++) {
  for(int r = 0; r < Panels::thePanels->panels[p].getHeight(); r++) {
  if(Panels::thePanels->panels[p].getPixel(c,r)->isTarget) {
  std::cou t << " TARGET IS " << p << " " << c << " " << r << " " << endl; 
  }
  }
  }
  }
  */
  oldTarget = -1;
  
  curTarget = random(12);
  
  holdTimeLeft = targetFlashHoldTime;
  
  
}

void AmbTargetFlash::updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {
  if (holdTimeLeft <= 0) {
    holdTimeLeft = targetFlashHoldTime;
    oldTarget = curTarget;
    curTarget = random(12);
    
  }
  
  
  if(curTarget != oldTarget ) {
    switch(curTarget) {
    case 1:
      new IGeneric(interps, addAmbientPixel(A, 6,8), flashAndFade, 1);
      //    Panels::thePanels->panels[A].getPixel(6,8)->addColor(255, 0, 0);
      break;
    case 2:
      new IGeneric(interps, addAmbientPixel(A, 9,1), flashAndFade, 1);
      //    Panels::thePanels->panels[A].getPixel(9,1)->addColor(255, 0, 0);
      break;
    case 3:
      new IGeneric(interps, addAmbientPixel(A, 12,5), flashAndFade, 1);
      //    Panels::thePanels->panels[A].getPixel(12, 5)->addColor(255, 0, 0);
      break;
    case 4:
      new IGeneric(interps, addAmbientPixel(A, 15,9), flashAndFade, 1);
      //    Panels::thePanels->panels[A].getPixel(15, 9)->addColor(255, 0, 0);
      break;
    case 5:
      new IGeneric(interps, addAmbientPixel(A, 18,4), flashAndFade, 1);
      //    Panels::thePanels->panels[A].getPixel(18, 4)->addColor(255, 0, 0);
      break;
    case 6:
      new IGeneric(interps, addAmbientPixel(B, 2,1), flashAndFade, 1);
      //    Panels::thePanels->panels[B].getPixel(2, 1)->addColor(255, 0, 0);
      break;
    case 7:
      new IGeneric(interps, addAmbientPixel(B, 5,4), flashAndFade, 1);
      //    Panels::thePanels->panels[B].getPixel(5, 4)->addColor(255, 0, 0);
      break;
    case 8:
      new IGeneric(interps, addAmbientPixel(C, 1,0), flashAndFade, 1);
      //    Panels::thePanels->panels[C].getPixel(1, 0)->addColor(255, 0, 0);
      break;
    case 9:
      new IGeneric(interps, addAmbientPixel(E, 0,5), flashAndFade, 1);
      //    Panels::thePanels->panels[E].getPixel(0, 5)->addColor(255, 0, 0);
      break;
    case 10:
      new IGeneric(interps, addAmbientPixel(I, 0,5), flashAndFade, 1);
      //    Panels::thePanels->panels[I].getPixel(0, 5)->addColor(255, 0, 0);
      break;
    case 11:
      new IGeneric(interps, addAmbientPixel(J, 0,2), flashAndFade, 1);
      //    Panels::thePanels->panels[J].getPixel(0, 2)->addColor(255, 0, 0);
      break;
      
    }
    oldTarget = curTarget;
  }
  
  holdTimeLeft -= dt;
  
}
