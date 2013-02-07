#ifndef __PATTERNPLUSSIGN_H__
#define __PATTERNPLUSSIGN_H__

#include "Pattern.h"
#include "AVPlusSign.h"

class PatternPlusSign : public Pattern {


public:


  virtual void setAvatars(PersonStats *ps, bool isEnter);

}
;

#endif