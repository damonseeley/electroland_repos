#include "PeopleStats.h"
#include "globals.h"
int PeopleStats::curAvatarGroup = 0;
PeopleStats *PeopleStats::thePeopleStats = NULL;

PeopleStats::PeopleStats(int hashSize) {
	if(thePeopleStats == NULL) {
		thePeopleStats = this;
	}
  curAvatarGroup = 0;
  size = hashSize;
  stats = new PersonStats*[size];
  for(int i = 0; i < size ; i++) {
    stats[i] = NULL;
  }
  headTot = NULL;
  transType = NOTRANS;
  isInTransition = false;
  peopleCnt = 0;
  interpGen = new InterpGen();
}




PeopleStats::~PeopleStats() {
 // c ut << "WARNING: PersonStats destructor incomplete" << endl;
}

void PeopleStats::add(PersonStats *ps) {
  ps->setPeopleStats(this);
  
  int hash = ps->hashcode;
  PersonStats* head = stats[hash];
  
  if (head == NULL) {
    ps->next = NULL;
    ps->prev = NULL;
    stats[hash] = ps;
  } else { // insert at head always
    ps->next = head;
    ps->prev = NULL;
    head->prev = ps;
    stats[hash] = ps;
  }
  
  if (headTot == NULL) {
    ps->nextTot = NULL;
    ps->prevTot = NULL;
    headTot = ps;
  } else { // insert at head always
    ps->nextTot = headTot;
    ps->prevTot = NULL;
    headTot->prevTot = ps;
    headTot = ps;
  }
  peopleCnt++;
}

void PeopleStats::display() {
  PersonStats *cur = headTot;
  while(cur) {
    cur->display();
    cur = cur->nextTot;
  }
}

// will return null if not there
PersonStats *PeopleStats::get(unsigned long id) {
  int hash = id % size;
  hash = (hash < 0) ? -hash : hash;
  PersonStats* stat = stats[hash];
  
  while((stat != NULL) && (stat->id != id)) {
    stat = stat->next;
  }
  return stat;
}

PersonStats *PeopleStats::getHead() {
	return headTot;
}

PersonStats *PeopleStats::getRandom() {
	int p = random(peopleCnt);
	PersonStats *cur = headTot;
	cout << "choosing " << p << "/" << peopleCnt;
	int i = 0;
    while(cur && (i++ < p)) {
		cur = cur->nextTot;
	}
	return cur;
}
PersonStats *PeopleStats::getSE() {
  PersonStats *se = headTot;
      PersonStats *cur = headTot;
    while(cur) {
      if (((cur->x) >= (se->x)) && ((cur->y) >= (se->y))) {
        se = cur;
      }
      cur = cur->nextTot;
    }
    return se;

  }

bool PeopleStats::remove(unsigned long id) {
  return remove(get(id));
}

bool PeopleStats::removeAndDestroy(unsigned long id) {
  return removeAndDestroy(get(id));
}
bool PeopleStats::remove(PersonStats *ps) {
	if (ps == NULL) return false;

	if (ps->prev == NULL) { // if head
		if (ps->next == NULL) { // if not only one
			stats[ps->hashcode] = NULL;
		} else {
			ps->next->prev = NULL;
			stats[ps->hashcode] = ps->next;
		}
	} else if (ps->next == NULL) { // if tail
		ps->prev->next = NULL;
	} else { // in the middle
		ps->prev->next = ps->next;
		ps->next->prev = ps->prev;
	}

	if (ps->prevTot == NULL) { // if head
		if (ps->nextTot == NULL) { // if not only one
			headTot = NULL;
		} else {
			ps->nextTot->prevTot = NULL;
			headTot = ps->nextTot;
		}
	} else if (ps->nextTot == NULL) { // if tail
		ps->prevTot->nextTot = NULL;
	} else { // in the middle
		ps->prevTot->nextTot = ps->nextTot;
		ps->nextTot->prevTot = ps->prevTot;
	}
	peopleCnt--;
	return true;
}



void PeopleStats::update(int curTime, int deltaTime) {
  
  
  if (transType == NOTRANS) {
    isInTransition = false;
    
    PersonStats *cur = headTot;
    while(cur) {
      cur->update(curTime, deltaTime);
      cur = cur->nextTot;
    }
  } else {
    float scale;
    if(interpGen->isRunning) {
      scale = interpGen->update(deltaTime);
    } else {
      interpGen->reset();
      scale = 1.0f;
    }

    float invScale = 1.0f - scale;
    if(transType == TOBLACK) {
      PersonStats *cur = headTot;
      while(cur) {
        cur->setAvatarGroupScale(oldAvatarGroup, invScale);
        cur->update(curTime, deltaTime);
        cur = cur->nextTot;
        
      }
    } else { // trans to other
      PersonStats *cur = headTot;
      while(cur) {
        cur->setAvatarGroupScale(oldAvatarGroup, invScale);
        cur->setAvatarGroupScale(curAvatarGroup, scale);
        cur->update(curTime, deltaTime);
        if (scale == 1.0) {
          cur->setAvatarGroupActivation(oldAvatarGroup, false);
//          c out << "activation false" << endl;
          
        }
        cur = cur->nextTot; 
        
      }
    }
    
    if (scale == 1.0) {
      transType = NOTRANS;
    }
    
  }
  /*
  
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
        }
        
  */
  
  
  /*
  
    PersonStats *cur = headTot;
    while(cur) {
    if (transType == TOBLACK) {
    cur->setAvatarGroupScale(oldAvatarGroup, otherScale);
    } else if  (transType == TOOTHER) {
    if (startTrans) {
				cur->setAvatarGroupActivation(curAvatarGroup, true); // will set to false by self when scale <= 0
        }
        cur->setAvatarGroupScale(oldAvatarGroup, otherScale);
        cur->setAvatarGroupScale(curAvatarGroup, curScale);
        }
        
          cur->update(curTime, deltaTime);
          cur = cur->nextTot;
          }
          
            
              if (transType != NOTRANS) {
              if (curScale > 1.0f) {
              transType = NOTRANS;
              } else {
              curScale +=dScalePerUsec * deltaTime;
              }
              if (startTrans) {
              startTrans = false;
              
                }
                }
  */
  
}

bool PeopleStats::removeAndDestroy(PersonStats *ps) {
  if(remove(ps)) {
	  ps->exitAvatars();
    delete ps;
    return true;
  } else {
    return false;
  }
  
}

void PeopleStats::transitionTo(Pattern *newPattern, int uSecs) {
  isInTransition = true;
  oldAvatarGroup = curAvatarGroup;
  curAvatarGroup++;
  curAvatarGroup = (curAvatarGroup >= AVATARGRPS) ? 0 : curAvatarGroup;
  
  Pattern *theOldPattern = Pattern::theCurPattern;
  
  if (newPattern == NULL) {
    transType = TOBLACK;
    Pattern::theCurPattern = Pattern::blankPattern;
    PersonStats *cur = headTot;
    while(cur) {
      cur->setAvatarGroupActivation(curAvatarGroup, true); // will set to false by self when scale <= 0
      cur = cur->nextTot;
    }
  } else {
    transType = TOOTHER;
    Pattern::theCurPattern = newPattern;
    PersonStats *cur = headTot;
    while(cur) {
      cur->setAvatarGroupActivation(curAvatarGroup, true); // will set to false by self when scale <= 0
      Pattern::theCurPattern->setAvatars(cur, false);
      cur = cur->nextTot;
    }
    
  }
  if(theOldPattern != NULL) {
	  if(theOldPattern  != Pattern::blankPattern) {
		  delete theOldPattern;
	  }
  }
  interpGen->start(uSecs);
  
  
}
