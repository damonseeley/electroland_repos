// Dummies.cpp: implementation of the Dummies class.
//
//////////////////////////////////////////////////////////////////////

#include "Dummies.h"


Dummies::Dummies(PeopleStats *p, WorldStats *w){
  people = p;
  world = w;
  for (int i = 0; i < DUMMYCNT; i++) {
    Dummy *d = &dummy[i];
    d->personId = -i;
    d->x = 6;
    d->y = -2;
    d->z = 5.5 * 12 * 2.54; // five foot six in cm;
    d->dx = 0;
    d->dy = 0;
  }
  lastDummy = 0;

//  tmpPat = PATA;

}

Dummies::~Dummies(){
}


void Dummies::genDummys(int c) {
  for(int i = 0; i < c; i++) {
    lastDummy++;
    lastDummy = (lastDummy >= DUMMYCNT) ? 0 : lastDummy;
    Dummy *d = &dummy[i];
    d->y = 0;
    d->x = random(500);
    d->dx = random(5) - 2;
    d->dy = random(10);
    d->personId = -lastDummy;
  }

  PersonStats *ps = people->get(- lastDummy);
  if (ps != NULL) {
    people->removeAndDestroy(ps);
    world->remove(ps);
  } 
  ps = new PersonStats(selected->personId, 0);
   Pattern::theCurPattern->setAvatars(ps, true);

  people->add(ps);
  world->add(ps);





}




bool  Dummies::setSelectMode(int i)  {
//  if (i == 0) {
 //   new MCA(people);
 // } else if (i == 1) {
 //   new MCB(people);
 // }
  /*
  if (i == 0) {
    if (tmpPat != PATA) {
      tmpPat = PATA;
      cou t << "trans a" << endl;
      people->transitionTo(new PatternA(), 2000);
    }

  } else if (i == 1) {
    if (tmpPat != PATB) {
      cou t << "trans b" << endl;
      tmpPat = PATB;
      people->transitionTo(new PatternB(), 2000);
    }
  }
 */
  if ((i >= 0) && (i < DUMMYCNT)) {
    selected = &dummy[i];
    return true;
  } else {
    selected = NULL;
    return false;
  }
  return true;
}

bool Dummies::inBounds(float x, float y) {
  if (x < 0) return false;
  if (x > roomMaxX) return false;
  if (y < 0) return false;
  if (y > roomMaxY) return false;

  return true;
}

void Dummies::update(float dx, float dy) {
  
  if (selected) {
    float x = selected->x;
    float y = selected->y;

    bool wasInBounds = inBounds(x,y);

    x+=dx;
    y+=dy;

    selected->x = x;
    selected->y = y;

    if (wasInBounds) {
      if(inBounds(x,y)) {
         PersonStats *ps = people->get(selected->personId);
          ps->update(x,y,selected->z);
         world->update(ps);
      } else {
        PersonStats *ps = people->get(selected->personId);
        world->remove(ps);
        people->removeAndDestroy(ps);
      }
    } else {
      if(inBounds(x,y)) { // enter

        PersonStats *ps = new PersonStats(selected->personId, 0);
           Pattern::theCurPattern->setAvatars(ps, true);

        people->add(ps);
        world->add(ps);
      } 
      // don't do anything if was outside and still outside
    }

  }
}
