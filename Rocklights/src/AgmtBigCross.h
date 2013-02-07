#ifndef __AGMTBIGCROSS_H__
#define __AGMTBOGCROSS_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtBigCross : public Arrangement {
public:

	static int red[]; 
	static int green[]; 
	static int blue[]; 

	AgmtBigCross(){};
	~AgmtBigCross(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "BigCross"; };
}
;

#endif