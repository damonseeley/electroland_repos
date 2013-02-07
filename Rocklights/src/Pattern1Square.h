#ifndef __PATTERN1SQUARE_H__
#define __PATTERN1SQUARE_H__

#include "Pattern.h"
#include "AV1Square.h"
#include "AV1SquarePulse.h"
#include "AV1SquareKit.h"
#include "AV1SquareBUS.h"
#include "AvatarCreatorHash.h"

class Pattern1Square : public Pattern {

  enum { MIXED, SINGLE, PULSE, KIT, BUS };

  int mode;
  int pilDir;
  int tailDelay;

public:

  Pattern1Square(int aveModem, int colDir, int trailLenth);

  virtual void setAvatars(PersonStats *ps, bool isEnter);

}
;

#endif