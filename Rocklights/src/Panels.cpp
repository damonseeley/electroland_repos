/*
 *  Panels.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#include "Panels.h"

Panels *Panels::thePanels = NULL; // there can be only one

Panels::Panels(float halfW, float halfH, float r) {
  if (thePanels != NULL) {
    timeStamp(); clog << "ERROR  Attemp to create 2nd panel object" << endl;
			Globals::hasError = true;

    delete this;
    return;
  }
  thePanels = this;
  
  halfWidth = halfW;
  halfHeight = halfH;
  targetRadius = r;
  
}

void Panels::setPixel(int let, int col, int row, BasePixel *pixel) {
//  std::co ut << "Panels::setPixel " << let << " " << col << "," << row << std::endl;
  panels[let].setPixel(col, row, pixel);
}

void Panels::display() {
  for (int i = 0; i < PANEL_CNT; i++) {
    panels[i].display();
  }
}

void Panels::topDisplay() {
  for (int i = 0; i < PANEL_CNT; i++) {
    panels[i].topDisplay();
  }
}

void Panels::update() {
  for (int i = 0; i < PANEL_CNT; i++) {
    panels[i].update();
  }
}
/*
void Panels::print() {
  cout << "A: " << getWidth(Panels::A) << "x" << getHeight(Panels::A) << endl;
  cout << "B: " << getWidth(Panels::B) << "x" << getHeight(Panels::B) << endl;
  cout << "C: " << getWidth(Panels::C) << "x" << getHeight(Panels::C) << endl;
  cout << "D: " << getWidth(Panels::D) << "x" << getHeight(Panels::D) << endl;
  cout << "E: " << getWidth(Panels::E) << "x" << getHeight(Panels::E) << endl;
  cout << "F: " << getWidth(Panels::F) << "x" << getHeight(Panels::G) << endl;
  cout << "G: " << getWidth(Panels::G) << "x" << getHeight(Panels::H) << endl;
  cout << "H: " << getWidth(Panels::H) << "x" << getHeight(Panels::J) << endl;
  cout << "I: " << getWidth(Panels::I) << "x" << getHeight(Panels::I) << endl;
  cout << "J: " << getWidth(Panels::J) << "x" << getHeight(Panels::J) << endl;
  
}
*/
void Panels::setLights(LightFile *lightFile, DataEnabler dataEnablers[], bool isTarget) {
  if(! isTarget) {
    for(int i = 0; i < PANEL_CNT; i++) {
      panels[i].set(lightFile->getWidth(i), lightFile->getHeight(i));
    }
  }
  
  char c;
  int col;
  int row;
  float x;
  float y;
  float z;
  int de;
  int ch;
  
  while(lightFile->readLine(c, col, row, x, y, z, de, ch)) {
    if(isTarget) {
      if (ch >= dataEnablers[de].size) {
        timeStamp(); clog << "ERROR  Error in configuration files.  Light DMX channel " << ch << " is greater than DataEnabler's " << de << " size in for light " << (char) c << " " << col << ", " << row << std::endl;
				Globals::hasError = true;

      }
    } else {
      if (ch+12 > dataEnablers[de].size) {
        timeStamp(); clog << "ERROR  Error in configuration files.  Light DMX channel " << ch << " is greater than DataEnabler's " << de << " size in for light " << (char) c << " " << col << ", " << row << std::endl;
				Globals::hasError = true;

      }
    }

    BasePixel *p=NULL;
    
    if(isTarget) {
      LightElement *le1 = new LETargetCircle(dataEnablers[de].data, ch);
      p =   new TargetPixel(le1);
      p->setDims(-targetRadius, -targetRadius, targetRadius, targetRadius);
    } else {
      LightElement *le1 = new LECoveStick(dataEnablers[de].data, ch);
      LightElement *le2 = new LECoveStick(dataEnablers[de].data, ch + 3);
      LightElement *le3 = new LECoveStick(dataEnablers[de].data, ch + 6);
      LightElement *le4 = new LECoveStick(dataEnablers[de].data, ch + 9);
      p =  new Pixel(le1, le2, le3, le4);
      p->setDims(-halfHeight, -halfWidth, halfHeight, halfWidth);
    }
    
    p->setPos(x,y,z);
    switch(c-'A') {
      case A:
        p->setTopPos(x, y, 0);
        break;
      case B:
        p->setTopPos(- z, y, 0);        
        p->setRot(90.0f, 0.0f, 1.0f, 0.0f);
        break;
      case C:
      case D:
        p->setTopPos(x, y+z, 0);        
        p->setRot(120.0f, 1.0f, 1.0f, -1.0f);
        p->setTopRot(90.0f, 0.0f, 0.0f, 1.0f);
        
        break;
      case E:
        p->setTopPos(x, y+z + 24, 0);        
        p->setRot(120.0f, 1.0f, 1.0f, -1.0f);
        p->setTopRot(90.0f, 0.0f, 0.0f, 1.0f);
        break;
      case F:
      case G:
	     p->setTopPos(- z, y, 0);        
        p->setRot(90.0f, 0.0f, 1.0f, 0.0f);
	//	p->setTopPos(x + z, y, 0);        
     //   p->setRot(120, 1.0f,1.0f,-1.0f);
        break;
      case H:
        p->setTopPos(x, y-z, 0);        
        p->setRot(120.0f, 1.0f, 1.0f, -1.0f);
        p->setTopRot(90.0f, 0.0f, 0.0f, 1.0f);
        break;
      case I:
      case J:
        p->setTopPos(x, y-z, 0);        
        p->setRot(120, 1.0f,1.0f,-1.0f);
        p->setTopRot(90.0f, 0.0f, 0.0f, 1.0f);
        break;
    }
    setPixel(c-'A', col-lightFile->getMinCol(c-'A'), row-lightFile->getMinRow(c-'A'), p);
  }
}

BasePixel* Panels::getPixel(int let, int col, int row) {
  return panels[let].getPixel(col, row);
}


BasePixel* Panels::getPixel(int let, int col) {
  return panels[let].getPixel(col);
}


