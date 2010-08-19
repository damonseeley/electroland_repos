#ifndef __PROJ_CELL_H__
#define __PROJ_CELL_H__

#include "limits.h"

struct ProjCell {
	ProjCell() {
		min = INT_MAX;
		max = 0;
		total = 0.0f;
		cnt = 0;
	}
	int min;
	int max;
	float total; 
	unsigned int cnt;
}
;

#endif