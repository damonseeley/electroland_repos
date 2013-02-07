#include "AmbientPixel.h"


  AmbientPixel::AmbientPixel(Panel *p, int col, int row, int stick)  {
      user = NULL;
  r = g= b= 0;
  scale =1.0f;

  if (stick <= -1) {
    pixel = p->getPixel(col, row);
  } else {
    pixel = p->getPixel(col, row)->getSubPixel(stick);
  }

  pixelType = AMBIENT;
  
  }

  AmbientPixel::~AmbientPixel() {
    clear();
    if(user) user->notifyPixelDeletion(this);
  }

  void AmbientPixel::updateScale(float avatarScale) {
    scale = avatarScale;
  }


  bool AmbientPixel::setScale(float s) {
      scale = s;
      return true;

  }


  void AmbientPixel::addColor(int r, int g, int b) { 
    pixel->addColor(r * scale, g*scale, b*scale); 
  }

   void AmbientPixel::addColor(int subPixel, int r, int g, int b) {
      pixel->addColor(subPixel, r*scale, g*scale, b*scale); 
    
    }
