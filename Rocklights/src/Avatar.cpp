/*
 *  Avatar .cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#include "Avatar.h"
#include "IGeneric.h"


int Avatar::pilInerp[3][7][25] = 
{ { { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},// 1
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//2
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//3
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//4
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//5
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//6
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} }, //7 ,
  { { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},// 1
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//2
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//3
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//4
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//5
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//6
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} }, //7 ,
  { { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},// 1
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//2
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//3
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//4
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//5
    { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25},//6
{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25} } } //7 ,
;

int Avatar::tailFade[3][8] =  { {1,2,3,4,5,6,7,8 } ,{1,2,3,4,5,6,7,8 }, {1,2,3,4,5,6,7,8}};

int Avatar::defTailDelay = 1000;

bool Avatar::staticInit = false;

  int Avatar::pillarLightSpeed = -1;

  Avatar::Avatar() {
	  unEntered = false;
	  name = "";
	  enterArrangement = NULL;
	  exitArrangement = NULL;
	  moveArrangement = NULL;
	  overheadArrangement = NULL;
	  enterSound = "";
	  exitSound = "";
	  moveSound = "";
	  enterSoundLoop = 1;

    c0 = NULL;
	curOffsetPixelVec = &offsetPixelVec1;
	oldOffsetPixelVec = &offsetPixelVec2;
    if(!staticInit) {
      staticInit = true;
      pillarLightSpeed = CProfile::theProfile->Int("pillarLightSpeed", 50);
      defTailDelay = CProfile::theProfile->Int("avatarTailHold", 750);

      for(int color = RED; color <= BLUE; color++) {
        int r = (color == RED) ? 255 : 0;
        int g = (color == GREEN) ? 255 : 0;
        int b = (color == BLUE) ? 255 : 0;

        pilInerp[color][0][0] = 0;
        pilInerp[color][0][1] = 0;
        pilInerp[color][0][2] = 0;

        pilInerp[color][0][3] = r;
        pilInerp[color][0][4] = g;
        pilInerp[color][0][5] = b;

        pilInerp[color][0][6] = pillarLightSpeed;


        pilInerp[color][0][7] = r;
        pilInerp[color][0][8] = g;
        pilInerp[color][0][9] = b;

        pilInerp[color][0][10] = r;
        pilInerp[color][0][11] = g;
        pilInerp[color][0][12] = b;

        pilInerp[color][0][13] = INT_MAX;

        pilInerp[color][0][14] = -1;

        for (int h = 1; h < 7; h++) {
        pilInerp[color][h][0] = 0;
        pilInerp[color][h][1] = 0;
        pilInerp[color][h][2] = 0;

        pilInerp[color][h][3] = 0;
        pilInerp[color][h][4] = 0;
        pilInerp[color][h][5] = 0;

        pilInerp[color][h][6] = pillarLightSpeed * h;

        pilInerp[color][h][7] = 0;
        pilInerp[color][h][8] = 0;
        pilInerp[color][h][9] = 0;

        pilInerp[color][h][10] = r;
        pilInerp[color][h][11] = g;
        pilInerp[color][h][12] = b;

        pilInerp[color][h][13] = pillarLightSpeed;

        pilInerp[color][h][14] = r;
        pilInerp[color][h][15] = g;
        pilInerp[color][h][16] = b;

        pilInerp[color][h][17] = r;
        pilInerp[color][h][18] = g;
        pilInerp[color][h][19] = b;

        pilInerp[color][h][20] = INT_MAX;

        pilInerp[color][h][21] = -1;
        }

        /*
    tailFade[color][0] = r;
    tailFade[color][1] = g;
    tailFade[color][2] = b;

    tailFade[color][3] = 0;
    tailFade[color][4] = 0;
    tailFade[color][5] = 0;

    tailFade[color][7] = 500;
    tailFade[color][8] = -1;
*/
        
   

    tailFade[color][0] = r;
    tailFade[color][1] = g;
    tailFade[color][2] = b;

    tailFade[color][3] = 0;
    tailFade[color][4] = 0;
    tailFade[color][5] = 0;

    tailFade[color][6] = 600;
    tailFade[color][7] = -1;

    }

    }

    //offsetPixelCnt = 0; 
	scale = 1.0f; panels = Panels::thePanels;
      oldCol = -1;
  oldCol = -1;
  tailDelay = 0;
  pillarMode = 1;
  oldPil = 0;
  renderPerdictorLength = -1;
  tailDelay = defTailDelay;  
  avColor = 1;


  }

void Avatar::update(PersonStats *personStats, int ct, int dt, Interpolators *interps) {

  row = personStats->row;
  col = personStats->col;

  if((row != oldRow) || (col != oldCol)) {
	  move(col, row, interps);
  }


  updateOffsetPixels(personStats, dt);
  
  
  updateFrame(personStats, ct, dt, interps);

   

  renderPillars(personStats, interps);
//  renderTail(personStats, interps);
//  renderPerdictor(personStats, interps);

  oldRow = personStats->row;
  oldCol = personStats->col;
  
}



void Avatar::setTrailDelay(int ms) {
  tailDelay = ms;
//  tailFade[RED][6] = tailDelay;
//  tailFade[GREEN][6] = tailDelay;
 // tailFade[BLUE][6] = tailDelay;

}

void Avatar::setPillarMode(int dir) {
  pillarMode = dir;
}



void Avatar::renderPerdictor(PersonStats *personStats, Interpolators *interps) {
  /*
  if(renderPerdictorLength > 0) {
    if ((col != oldCol) || (row != oldRow)) {
      int c = col - oldCol;
      int r = row - oldRow;

      if (c < 0) {
        c = - 1;
      } else if (c == 0) {
        c = 0;
      } else {
        c = 1;
      }
      if (r < 0) {
        r =  - 1;
      } else if (r == 0) {
        r = 0;
      } else {
        r =  +1;
      }

      int cc = col ;
      int cr = row ;

      for(int i = 0; i < 3; i++) {
        cc+=c;
        cr+=r;

        new IGeneric(interps, panels->getPixel(A, cc, cr), redPredict, 1);
      }
    }
  }
*/
}

void Avatar::renderTail(PersonStats *personStats, Interpolators *interps) {
  if (tailDelay > 0) { 
    
    // this is a trail (leave something to fade out if row and column change)
    if ((col != oldCol) || (row != oldRow)) {
      new IGeneric(interps, panels->getPixel(A, oldCol, oldRow),tailFade[avColor]);
    }
  }
}
void Avatar::renderPillars(PersonStats *personStats, Interpolators *interps) {
  if (pillarMode == 0) return;

    // this turns on pillars when one is close to them
    int pil = personStats->nearPillar;
      if (pil != oldPil) {
        if(c0) {
			c0->needsReaping = true;
			c1->needsReaping = true;
			c2->needsReaping = true;
			c3->needsReaping = true;
			c4->needsReaping = true;
			c5->needsReaping = true;
			c6->needsReaping = true;
          c0 = NULL;
        }
    
    if (pil != 0) {
      if (pil != oldPil) {

        int bRow;

        int tPil = pil;
        if (tPil < 0) {
          bRow = -tPil;
          bRow -=1;
          tPil = 1;
        } else {
          bRow = 0;
        }
        if (pillarMode > 0) {        
          c0 = new IGeneric(interps, panels->getPixel(tPil, 0, bRow), pilInerp[avColor][6], 1);
          c1 = new IGeneric(interps, panels->getPixel(tPil, 1, bRow), pilInerp[avColor][5], 1);
          c2 = new IGeneric(interps, panels->getPixel(tPil, 2, bRow), pilInerp[avColor][4], 1);
          c3 = new IGeneric(interps, panels->getPixel(tPil, 3, bRow), pilInerp[avColor][3], 1);
          c4 = new IGeneric(interps, panels->getPixel(tPil, 4, bRow), pilInerp[avColor][2], 1);
          c5 = new IGeneric(interps, panels->getPixel(tPil, 5, bRow), pilInerp[avColor][1], 1);
          c6 = new IGeneric(interps, panels->getPixel(tPil, 6, bRow), pilInerp[avColor][0], 1);
        } else {
          c0 = new IGeneric(interps, panels->getPixel(tPil, 0, bRow), pilInerp[avColor][0], 1);
          c1 = new IGeneric(interps, panels->getPixel(tPil, 1, bRow), pilInerp[avColor][1], 1);
          c2 = new IGeneric(interps, panels->getPixel(tPil, 2, bRow), pilInerp[avColor][2], 1);
          c3 = new IGeneric(interps, panels->getPixel(tPil, 3, bRow), pilInerp[avColor][3], 1);
          c4 = new IGeneric(interps, panels->getPixel(tPil, 4, bRow), pilInerp[avColor][4], 1);
          c5 = new IGeneric(interps, panels->getPixel(tPil, 5, bRow), pilInerp[avColor][5], 1);
          c6 = new IGeneric(interps, panels->getPixel(tPil, 6, bRow), pilInerp[avColor][6], 1);
        }
        }
      }
    oldPil = pil;

    } 

}
void Avatar ::clear() {
//  for(int i = 0 ; i < offsetPixelCnt; i++) {
//    offsetPixel[i]->clear();
//  }
}


OffsetPixel* Avatar ::addOffsetPixel(int panelName, int c, int r, bool smooth) {
    Panel *panel = &(Panels::thePanels->panels[panelName]);
    OffsetPixel *p = new OffsetPixel(panel, c, r, smooth);
	curOffsetPixelVec->push_back(p);
//	cout << "       addOffsetPixel " << curOffsetPixelVec->size() << endl;
    return p;
}

OffsetPixel* Avatar ::addOffsetPixel(int panelName, int c, int r, int stick, bool smooth) {
    Panel *panel = &(Panels::thePanels->panels[panelName]);
    OffsetPixel *p = new OffsetPixel(panel, c, r, stick, smooth);
	curOffsetPixelVec->push_back(p);
//	cout << "       addOffsetPixel " << curOffsetPixelVec->size() << endl;
    return p;
}


void Avatar ::updateOffsetPixels(PersonStats *personStats, float dt) {
	while(curOffsetPixelVec->size() > 0) {
		OffsetPixel* op = curOffsetPixelVec->back();
		curOffsetPixelVec->pop_back();
		op->updatePosition(personStats, dt, scale);
		if (op->needsReaping) {
//			op->println();
			delete op;
		} else {
			oldOffsetPixelVec->push_back(op);
		}
	}

	vector<OffsetPixel *> *tmp = curOffsetPixelVec;
	curOffsetPixelVec = oldOffsetPixelVec;
	oldOffsetPixelVec = tmp;
	oldOffsetPixelVec->clear();



}

Avatar::~Avatar() {
	while(curOffsetPixelVec->size() > 0) {
		OffsetPixel *op = curOffsetPixelVec->back();
		curOffsetPixelVec->pop_back();
		delete op;
	}
  if(c0) {
			c0->needsReaping = true;
			c1->needsReaping = true;
			c2->needsReaping = true;
			c3->needsReaping = true;
			c4->needsReaping = true;
			c5->needsReaping = true;
			c6->needsReaping = true;
  c0 = NULL;
  }
}

void Avatar::addColor(int panel, int col, int row, int r, int g, int b) {
  panels->panels[panel].getPixel(col, row)->addColor(scale * r,scale * g,scale * b);
}

void Avatar::addColor(int panel, int col, int row, int stick, int r, int g, int b) {
  panels->panels[panel].getPixel(col, row)->getSubPixel(stick)->addColor(scale * r,scale * g,scale * b);
}
