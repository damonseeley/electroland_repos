#ifndef __AGMTO_H__
#define __AGMTO_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtO : public Arrangement {
public:

//	static int red[]; 

	AgmtO(){};
	~AgmtO(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "O"; };
}
;

#endif