#ifndef __PATTERNPIXELDANCE_H__
#define __PATTERNPIXELDANCE_H__

#include "Pattern.h"
#include "AVSinglePixelDance.h"

class PatternPixelDance : public Pattern {


public:


  virtual void setAvatars(PersonStats *ps, bool isEnter);

}
;

#endif