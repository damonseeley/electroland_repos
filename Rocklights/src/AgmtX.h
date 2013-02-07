#ifndef __AGMTX_H__
#define __AGMTX_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtX : public Arrangement {
public:

//	static int red[]; 

	AgmtX(){};
	~AgmtX(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "X"; };
}
;

#endif