#include "arrangementhash.h"
#include "AmbientCreatorHash.h"
#include "AgmtFlashBlock.h"
#include "Agmt1SquareRed.h"
#include "Agmt1SquareGreen.h"
#include "Agmt1SquareBlue.h"
#include "Agmt1SquareRedKO.h"
#include "Agmt1SquareGreenKO.h"
#include "Agmt1SquareBlueKO.h"
#include "Agmt1SquareKO.h"
#include "AgmtVelScale.h"
#include "AgmtPlus.h"
#include "AgmtSpinner.h"
#include "AgmtBlueLineKO.h"
#include "AgmtO.h"
#include "AgmtX.h"
#include "AgmtSinglePixelDance.h"
#include "AgmtSquarePulse.h"
#include "AgmtWashEnter.h"
#include "AgmtBigCross.h"

ArrangementHash *ArrangementHash::theHash = NULL;

ArrangementHash::ArrangementHash(void)
{
	if (theHash == NULL) {
		theHash = this;
		theHash->add(new Agmt1Square());
		theHash->add(new Agmt1SquareRed());
		theHash->add(new Agmt1SquareBlue());
		theHash->add(new Agmt1SquareGreen());
		theHash->add(new AgmtFlashBlock());
		theHash->add(new Agmt1SquareRedKO());
		theHash->add(new Agmt1SquareGreenKO());
		theHash->add(new Agmt1SquareBlueKO());
		theHash->add(new Agmt1SquareKO());
		theHash->add(new AgmtVelScale());
		theHash->add(new AgmtPlus());
		theHash->add(new AgmtSpinner());
		theHash->add(new AgmtBlueLineKO());
		theHash->add(new AgmtBlueLineKOV());
		theHash->add(new AgmtO());
		theHash->add(new AgmtX());
		theHash->add(new AgmtSinglePixelDance());
		theHash->add(new AgmtSquarePulse());
		theHash->add(new AgmtWashEnter());
		theHash->add(new AgmtBigCross());
	}
}

ArrangementHash::~ArrangementHash(void)
{
}

void ArrangementHash::add(Arrangement* a) {
	Arrangement* dup = get(a->getName());
	if(dup == NULL) {
		hash.insert(entry(a->getName(), a));
		cout << "Pats: " << a->getName() << "\n";
	} else {
		timeStamp(); clog << "WARNING  Attempt to add AmbientCreator wilth duplicate name " << dup->getName() << "\n";
	}
}

Arrangement* ArrangementHash::get(string name) {
		hash_map<string,Arrangement*>::iterator it;
		it = hash.find(name);
		if(it != hash.end()) return it->second;
		return NULL; // if not found
}

void ArrangementHash::apply(string name, Avatar *a, Interpolators *interps) {
	Arrangement* at = get(name);
	if (at != NULL) {
		at->apply(a, interps);
	} else {
		timeStamp(); clog << "WARNING  Arrangement not found for name " << name << "\n";
	}
}