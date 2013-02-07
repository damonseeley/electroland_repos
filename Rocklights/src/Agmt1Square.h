#ifndef __AGMT1SQUARE_H__
#define __AGMT1SQUARE_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class Agmt1Square : public Arrangement {
public:
	static int red[];
	static int green[];
	static int blue[];
  static int redFade[];
  int oldRow;
  int oldCol;
  int oldPil;



  static int redIn6[];
  static int redIn5[];
  static int redIn4[];
  static int redIn3[];
  static int redIn2[];
  static int redIn1[];
  static int redIn0[];

  IGeneric *c0;
  IGeneric *c1;
  IGeneric *c2;
  IGeneric *c3;
  IGeneric *c4;
  IGeneric *c5;
  IGeneric *c6;

	Agmt1Square(){};
	~Agmt1Square(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "Square"; };
}
;

#endif