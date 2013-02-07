#ifndef __ARRANGEMENTHASH_H__
#define __ARRANGEMENTHASH_H__

#include <string>
#include <hash_map>
#include "globals.h"
#include "Arrangement.h"
#include "SoundHash.h"
#include "Agmt1Square.h"

class ArrangementHash {

public:
	static ArrangementHash *theHash;
	typedef pair<string,Arrangement*> entry;
	hash_map<string,Arrangement*> hash;

public:
	ArrangementHash();
	~ArrangementHash();
	void add(Arrangement* ac);
	Arrangement* get(string name);
	void apply(string name, Avatar *a, Interpolators *interps);
}
;

#endif
