/*
 *  Interpolators.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#include "Interpolators.h"
#include "IGeneric.h"
#include <vector>

//Interpolators *Interpolators::theInterpolators = NULL;

Interpolators::Interpolators() {
	curInterps = &interps1;
	oldInterps = &interps2;
}



Interpolators::~Interpolators() {
	while(curInterps->size() > 0) {
		Interpolator *interp = curInterps->back();
		curInterps->pop_back();
		delete interp;
	}
}


void Interpolators::add(Interpolator *interp) {
	curInterps->push_back(interp);
}

void Interpolators::remove(Interpolator *interp) {
	interp->needsReaping = true;
}


void Interpolators::update(int curTime, int deltaTime, float scale) {
	while(curInterps->size() > 0) {
		Interpolator *interp = curInterps->back();
		curInterps->pop_back();
		if(interp->needsReaping) {
			delete interp;
		} else {
			interp->update(curTime, deltaTime, scale);
			oldInterps->push_back(interp);
		}
	}
	vector<Interpolator *> *tmp = curInterps;
	curInterps = oldInterps;
	oldInterps = tmp;
}