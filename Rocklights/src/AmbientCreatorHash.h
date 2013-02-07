#ifndef __AMBIENTCREATERHASH_H__
#define __AMBIENTCREATERHASH_H__

#include <string>
#include <hash_map>
#include "globals.h"
#include "AmbientCreator.h"
#include "AmbEmpty.h"
#include "AmbRedStick.h"
#include "SoundHash.h"


//	size_t std::hash_compare<std::string>::operator ()(const std::string& s) const;
/*
{
	size_t h = 0;
    std::string::const_iterator p, p_end;
    for(p = s.begin(), p_end = s.end(); p != p_end; ++p)
    {
      h = 31 * h + (*p);
    }
    return h;
};
*/

class AmbientCreatorHash {

public:
	static AmbientCreatorHash *theACHash;
	static AmbientCreator* defaultCreator;
	typedef pair<string,AmbientCreator*> entry;
	hash_map<string,AmbientCreator*> creators;

public:
	AmbientCreatorHash();
	~AmbientCreatorHash();
	void add(AmbientCreator* ac);
	AmbientCreator* get(string name);
	Ambient* create(string name, bool createInterps = true);
}
;

#endif
