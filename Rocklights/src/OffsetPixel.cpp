#include "OffsetPixel.h"

//int OffsetPixel::idCnt = 0; // UNDONE remove

OffsetPixel::OffsetPixel(Panel *p, int cOffset, int rOffset, bool smooth) : BasePixel() { 
	needsReaping = false;
  isSmooth = smooth;

  panel = p;
  colOffset = cOffset;
  rowOffset = rOffset; 

  newPixel = BasePixel::blank;
  oldPixel = BasePixel::blank;

  scale = 1.0f;

  interp = new InterpGen();

  if (crossFadeTime < 0) {
    crossFadeTime = CProfile::theProfile->Int("pixelCrossFadeTime", 100);
  }
  pixelType = OFFSET;
  
//  id = idCnt++;
//  cou t << id << " created" << endl;


  stick = -1;
}

OffsetPixel::OffsetPixel(Panel *p, int cOffset, int rOffset, int st, bool smooth)  {
	needsReaping = false;
  isSmooth = smooth;

  panel = p;
  colOffset = cOffset;
  rowOffset = rOffset; 

  newPixel = BasePixel::blank;
  oldPixel = BasePixel::blank;

  scale = 1.0f;

  interp = new InterpGen();

  if (crossFadeTime < 0) {
    crossFadeTime = CProfile::theProfile->Int("pixelCrossFadeTime", 100);
  }
  pixelType = OFFSET;
  
//  id = idCnt++;
//  co ut << id << " created" << endl;
  
  stick = st;

}


OffsetPixel::~OffsetPixel() {
	needsReaping = true;
  clear();
  if (user)  user->notifyPixelDeletion(this);
//  co ut << "offset pixel " << colOffset << ", " << rowOffset << " deleted\n";
  // delete below with c out


  if(interp) {
    delete interp; // UNDONE was crashing here don't know why
    interp = NULL;
  }
//  cou t << id << " deleted" << endl;
}

int OffsetPixel::crossFadeTime = -1;


void OffsetPixel::updatePosition(PersonStats *ps, int deltaT, float avatarScale) {
  clear();


  int col = ps->col;
  int row = ps->row;

  int colO = col + colOffset;
  int rowO = row + rowOffset;

  newPixel = panel->getPixel(colO, rowO);

  if (isSmooth) {
    if (newPixel != oldPixel) {
      if(interp->wasStarted) {
        if(interp->isRunning) {
          scale = interp->update(deltaT);
        } else {
          interp->reset();
          oldPixel = newPixel;
          scale = 1.0f;
        }
      } else {
        interp->start(crossFadeTime);
        scale = 0.0f;
      }
    } else {
      scale = 1.0f;
     }

    scale *= avatarScale;
    scaleInv = 1.0 - scale;

  } else {
     newPixel = panel->getPixel(colO, rowO);
     scale = avatarScale;

  }


}
  
  


void OffsetPixel::addColor(int r, int g, int b) {
  if(stick >= 0) {
    addColor(stick, r, g, b);
  } else {
  if (isSmooth && (newPixel != oldPixel)) {
//    cou t << id <<" -- offsetPixel scale " << scale <<   "    inv " << scaleInv <<"\n";  FIX
    newPixel->addColor(scale * r,scale * g,scale * b);
    oldPixel->addColor(scaleInv * r,scaleInv * g,scaleInv * b);
  } else {
    newPixel->addColor(scale * r,scale * g,scale * b);
  }
  }
}

void OffsetPixel::addColor(int subPixel, int r, int g, int b) {
  if (isSmooth && (newPixel != oldPixel)) {
    newPixel->addColor(subPixel, scale * r,scale * g,scale * b);
    oldPixel->addColor(subPixel, scaleInv * r,scaleInv * g,scaleInv * b);
  } else {
    newPixel->addColor(subPixel, scale * r,scale * g,scale * b);
  }
}
