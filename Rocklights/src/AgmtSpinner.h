#ifndef __AGMTSPINNER_H__
#define __AGMTSPINNER_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtSpinner : public Arrangement {
public:

	static int spinnerBlue[];

	AgmtSpinner(){};
	~AgmtSpinner(){};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "Spinner"; };
}
;

#endif