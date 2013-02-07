#ifndef __AVATARCREATERHASH_H__
#define __AVATARCREATERHASH_H__

#include <string>
#include <hash_map>
#include "AvatarCreator.h"
#include "globals.h"
#include "AV1Square.h"
#include "AV9Square.h"
#include "AVGeneric.h"

/*
	size_t std::hash_compare<std::string>::operator ()(const std::string& s) const
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
class AvatarCreatorHash {

public:
	static AvatarCreator* defaultCreator;
	static AvatarCreatorHash *theACHash;
	typedef pair<string,AvatarCreator*> entry;
	hash_map<string,AvatarCreator*> creators;

public:
	AvatarCreatorHash();
	~AvatarCreatorHash();
	void add(AvatarCreator* ac);
	AvatarCreator* get(string name);

	Avatar* create(string name, PersonStats *personStats,  Interpolators *interps);
};

#endif
