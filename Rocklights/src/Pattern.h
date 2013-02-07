#ifndef __PATTERN_H__
#define __PATTERN_H__

#include "PersonStats.h"
#include "PeopleStats.h"

class PersonStats;
class PeopleStats;

class Pattern {
  // pointer to the current pattern
  // this is the pattern that tracker should call when there is a new person


public:
static bool inited;
  static Pattern *theCurPattern;
  static Pattern *blankPattern;

  Pattern();
  virtual ~Pattern(){};

  virtual void setAvatars(PersonStats *ps, bool isEnter);
  

}
;

#endif