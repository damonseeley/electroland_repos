/*
*	PersonStats.cpp
*	RockefellerCenter
*
*	Created by Eitan Mendelowitz on 9/30/05.
* 
*
*/

#include "PersonStats.h"
#include "Transformer.h"

float PersonStats::pilDistSquaredExit = -1.0f;
float PersonStats::pilDistSquaredEnter = -1.0f;

int PersonStats::nextColor = Avatar::RED;

PersonStats::PersonStats() {
  inpterps = new Interpolators();
  if (pilDistSquaredExit < 0) {
    pilDistSquaredEnter = CProfile::theProfile->Int("pilDistSquaredEnter", 10000);
    pilDistSquaredExit = CProfile::theProfile->Int("pilDistSquaredExit", 15000);
  }
  nearPillar = 0;
}


PersonStats::PersonStats(unsigned long iid, int curTime) {
  inited = false;
  exited = false;
  nearPillar = 0;
  
  id = iid;
  
  for(int i = 0; i < AVATARGRPS; i++) {
    avatarGroups[i].cnt = 0;
    avatarGroups[i].isActive = true;
    avatarGroups[i].scale = 1.0f;
  }
  
  
  enterTime = curTime;
  
  inpterps = new Interpolators();
  
  if (pixelStick < 0) {
    pixelStick = CProfile::theProfile->Float("pixelStickyness", .5f);
  }
  
  
  color = nextColor++;
  nextColor = (nextColor > Avatar::BLUE) ? Avatar::RED : nextColor;
  switch(color) {
	  case Avatar::RED:
		  r = .4f + (random(100) * .006f);
		  g = 0.0f;
		  b = 0.0f;
		  break;
	  case Avatar::GREEN:
		  r = 0.0f;
		  g = .4f + (random(100) * .006f);
		  b = 0.0f;
		  break;
	  case Avatar::BLUE:
		  r = 0.0f;
		  g = 0.0f;
		  b = .4f + (random(100) * .006f);
		  break;
  }
  
//  r = ((float) random(500)) * .002f;
//  g = ((float) random(500)) * .002f;
//  b = ((float) random(500)) * .002f;
  
  renderSize = (3.0f * 2.5f);
  
  if (pilDistSquaredExit < 0) {
    pilDistSquaredEnter = CProfile::theProfile->Int("pilDistSquaredEnter", 10000);
    pilDistSquaredExit = CProfile::theProfile->Int("pilDistSquaredExit", 15000);
  }
}

PersonStats::~PersonStats() {
  for(int i = 0; i < AVATARGRPS; i++) {
    clearAndDestroyAvatars(i);
  }
  
  //	delete[] &avatars;
}


void PersonStats::setPeopleStats(PeopleStats* psHash) {  
  hashtable = psHash;
  hashcode = id % hashtable->getSize();
  hashcode = (hashcode < 0) ? - hashcode : hashcode;
  
}


void PersonStats::update(int curTime, int deltaTime) {
  for(int i = 0; i < AVATARGRPS; i++) {
    if(avatarGroups[i].isActive) {
      for (int j = 0; j < avatarGroups[i].cnt; j++) {
        avatarGroups[i].avatar[j]->update(this, curTime, deltaTime, inpterps);
      }
      
      
    }
  }
  
  inpterps->update(curTime, deltaTime);
  wasUpdated =false;
  
}

void PersonStats::setAvatarGroupActivation(int i, bool b) {
  
  avatarGroups[i].isActive = b;
  if (!b) {
    clearAndDestroyAvatars(i);			
  }
  wasUpdated =true;    
}

void PersonStats::setAvatarGroupScale(int g, float s) {
  //	 co ut << "avatarGroups[" << g << "].scale = "<< s << endl;
  
  if (s >= 1.0f) {	
    avatarGroups[g].scale = 1.0f;
  } else {
    avatarGroups[g].scale = s;
  }
  
  if (avatarGroups[g].scale  < 0.0f) {
    avatarGroups[g].scale  = 0.0f;
  }
  for(int i = 0; i < avatarGroups[g].cnt; i++) {
    avatarGroups[g].avatar[i]->setScale(avatarGroups[g].scale);
  }
  
  
}

float PersonStats::pixelStick = -1.0;

void PersonStats::update(float fx, float fy, float fz) {
  if (! inited) {
	  x = fx;
	  y = fy;
	  Transformer::theTransformer->transform(x, y);
//	 cout << "p(" << x << ", " << y << ")" << endl;;
//    x = fx;
//    y = fy;
    h = fz;
    col = -1;
    row = -1;
    pixel = BasePixel::blank;
    inited = true;
    
  } 
		lastX = x;
    lastY = y;
    lastH = h;
    /*
    x *= .7f;
    y *= .7f;
    h *= .7f;
    
      x += fx * .3f;
      y += fy * .3f;
      h += fz * .3f;
    */
	  x = fx;
	  y = fy;
	  Transformer::theTransformer->transform(x, y);
//	 cout << "p(" << x << ", " << y << ")" << endl;

//    x = fx;
//    y = fy;
    h = fz;
    
    dX = x - lastX;
    dY = y - lastY;
    dH = h - lastH;
    
    
    
    int newCol = Panels::thePanels->panels[Panels::A].getCol(x);
    int newRow = Panels::thePanels->panels[Panels::A].getRow(y);
    
    BasePixel *newPixel =  Panels::thePanels->panels[Panels::A].getPixel(newCol, newRow);
    
    if (newPixel != pixel) {
      float gap;
      float dist;
      
      bool change = false;
      
      
      gap = newPixel->x - pixel->x;
      if (gap != 0.0f) {
        gap = (gap < 0.0f) ? - gap : gap;
        gap -= newPixel->halfWidth;
        gap -= pixel->halfWidth;
        
        dist = x - pixel->x;
        dist = (dist < 0.0f) ? - dist : dist;
        dist -= pixel->halfWidth;
        if ((dist/gap) > pixelStick) {
          change = true;
        }
      }
      if (! change) {
        gap = newPixel->y - pixel->y;
        if (gap != 0.0f) {
          gap = (gap < 0.0f) ? - gap : gap;
          gap -= newPixel->halfHeight;
          gap -= pixel->halfHeight;
          dist = y - pixel->y;
          dist = (dist < 0.0f) ? - dist : dist;
          dist -= pixel->halfWidth;
          if ((dist/gap) > pixelStick) {
            change = true;
            //				row = newRow;
          }
        }
      }
      
      if(change) {
        col  = newCol;
        row = newRow;
        pixel =  Panels::thePanels->panels[Panels::A].getPixel(col, row);
      }
      
    }
    calcNearPillar();
    
    wasUpdated = true;
    
    
}

void PersonStats::display() {
  //		glBegin(GL_LINE_LOOP);
  glBegin(GL_TRIANGLE_FAN);
		glColor4f(r, g, b, 0.4f); 	 
    glVertex3f(x, y, 0);		
    
    glVertex3f(x-renderSize, y-renderSize, PEOPLERENDERHEIGHT);			
    glVertex3f(x-renderSize, y+renderSize, PEOPLERENDERHEIGHT);			
    glVertex3f(x+renderSize, y+renderSize, PEOPLERENDERHEIGHT);			
    glVertex3f(x+renderSize, y-renderSize, PEOPLERENDERHEIGHT);			
    glVertex3f(x-renderSize, y-renderSize, PEOPLERENDERHEIGHT);	

    glEnd();					
	if(Globals::displayCoord) {
		sprintf(charBuf, "(%.1f, %.1f)", x, y);
		displayText(x +renderSize,y +renderSize,0, charBuf);
	}


}

void PersonStats::addAvatar(Avatar *a, int g) {
  if (avatarGroups[g].cnt < MAXAVATARS) {
    avatarGroups[g].avatar[avatarGroups[g].cnt] = a;
    avatarGroups[g].cnt++;
    a->setScale(avatarGroups[g].scale);
    // co ut << "adding with scale " << a->getScale() << endl;
    
  }
  
}

void PersonStats::exitAvatars() {  
	for(int g = 0; g < AVATARGRPS; g++) {
				if (avatarGroups[g].isActive) {
			for (int i = 0; i < avatarGroups[g].cnt ; i++) {
					avatarGroups[g].avatar[i]->exit(inpterps);
				}
		}
	}
    
}

void PersonStats::clearAndDestroyAvatars(int g) {
  avatarGroups[g].isActive = false;
  
		for (int i = 0; i < avatarGroups[g].cnt ; i++) {
			delete avatarGroups[g].avatar[i];
    }
    avatarGroups[g].cnt = 0;
    
}


// return 0 if not near any
// returns negative value if near back wall (value is -1 - negative closest row)
// return positive number for c-j
float PersonStats::distSquaredToPillar(int pil) {
  BasePixel *pix = Panels::thePanels->panels[pil].getPixel(0,0);
  float d = x - pix->x;
  d *=d;
  float tot = d;
  d = y - pix->y;
  d *= d;
  return tot + d;
}
void PersonStats::calcNearPillar() {
  if (nearPillar != 0) { // is was near pillar check for exit
    BasePixel *pix = NULL;
    if (nearPillar >= Panels::C) {
      if((nearPillar == Panels::E) || (nearPillar == Panels::F)) {
        if ((col >= 18) && (row <= 10)) {
          nearPillar = Panels::E;
        } else if((col < 18) && (row >= 10)) {
          nearPillar = Panels::F;
        }
      } else if ((nearPillar == Panels::G) || (nearPillar == Panels::H)) {
        if ((col >= 18) && (row >= 1)) {
          nearPillar = Panels::H;
        } else if((col < 18) && (row <= 1)) {
          nearPillar = Panels::G;
        }       
      }
      pix = Panels::thePanels->panels[nearPillar].getPixel(0,0);
    } else {
      if (row <= 3) {
        nearPillar = -1;
        pix = Panels::thePanels->panels[Panels::B].getPixel(0, 0);
      } else if (row >= 8) {
        nearPillar = -6;
        pix = Panels::thePanels->panels[Panels::B].getPixel(0, 5);
      } else {
        nearPillar = -row + 2;
        pix = Panels::thePanels->panels[Panels::B].getPixel(0, row - 3);
      }
    }
    float d = x - pix->x;
    d *=d;
    if(d > pilDistSquaredExit) {
      nearPillar = 0;
    } else {
      float tot = d;
      d = y - pix->y;
      d *= d;
      tot += d;
      if (tot > pilDistSquaredExit) {
        nearPillar = 0;
      }
    }
  } else {  
    int nearest = 0;
    float nearDist = pilDistSquaredEnter;
    for(int i = Panels::C; i < Panels::PANEL_CNT; i++) {
      BasePixel *pix = Panels::thePanels->panels[i].getPixel(0,0);
      float d = x - pix->x;
      d *=d;
      float tot = d;
      if(tot < nearDist) {
        d = y - pix->y;
        d *= d;
        tot += d;
        if (tot < nearDist) {
          nearDist = tot;
          nearest = i;
        }   
      }
    }
    
    if(nearest != 0) {
      nearPillar = nearest;
      return ;
    }
    
    
    Panel *panB = &Panels::thePanels->panels[Panels::B];
    BasePixel *panBPix = panB->getPixel(0,0);
    float dx = x - panBPix->x;
    if (x  <= panBPix->x) {
      nearPillar = 0;
      return ;
    }
    dx *= dx;
    if (dx >= pilDistSquaredEnter) {
      nearPillar = 0;
      return ;
    }
    
    float dy = y - panB->getPixel(0,r)->y;
    dy *= dy;
    float tot = dy + dx;
    if (tot < nearDist) {
      nearest = r + 1;
      nearDist = tot;
    }
    
    int bHeight = panB->getHeight();
    for (int r = 1 ; r < bHeight; r++) {
      dy = y - panB->getPixel(0,r)->y;
      dy *= dy;
      float tot = dy + dx;
      if (tot < nearDist) {
        nearest = r +1;
        nearDist = tot;
      }
    }
    
    if (nearest != 0) {
      nearPillar = -nearest;
    }
  }
  
}
void PersonStats::displayText(float x, float y, float z, const char* s) {
	glColor3f(r, g, b);
	glRasterPos3f(x, y, z);
	char c;
	int i = 0;
	while((c = s[i++])  != '\0') {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
	}
}
