#ifndef __PATTERNA_H__
#define __PATTERNA_H__

#include "Pattern.h"
#include "AVHuge.h"
#include "AV1Square.h"
class PatternA : public Pattern {


public:
  PatternA();

  virtual void setAvatars(PersonStats *ps, bool isEnter);

}
;

#endif