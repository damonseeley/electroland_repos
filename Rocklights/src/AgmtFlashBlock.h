#ifndef __AGMTFLASHBLOCK_H__
#define __AGMTFLASHBLOCK_H__

#include <string>
#include "Arrangement.h"
#include "IGeneric.h"

class AgmtFlashBlock : public Arrangement {
public:
	static int* flash; 

	AgmtFlashBlock();
	~AgmtFlashBlock() {};
	virtual void apply(Avatar *avater, Interpolators *interps);
	virtual string getName() { return "FlashBlock"; };
	void exit(int col, int row, Interpolators *interps);

}
;

#endif