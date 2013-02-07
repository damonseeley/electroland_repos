#ifndef __SOUNDHASH_H__
#define __SOUNDHASH_H__

#include <string>
#include <hash_map>
#include <iostream>


#include"al.h"
#include"alc.h"
#include"alut.h"

#include "Sound.h"
#include "Globals.h"
#include "SoundSourcePool.h"
using namespace stdext;
using namespace std;

	size_t stdext::hash_compare<std::string>::operator ()(const std::string& s) const
{
	size_t h = 0;
    std::string::const_iterator p, p_end;
    for(p = s.begin(), p_end = s.end(); p != p_end; ++p)
    {
      h = 31 * h + (*p);
    }
    return h;
};
  
class SoundHash {

public:
	bool useSound;
	static SoundHash *theSoundHash;
	typedef pair<string,Sound*> entry;
	hash_map<string,Sound*> sounds;
	SoundSourcePool ssp;
	static ALCdevice *g_currentDevice;
	static ALCcontext *g_currentContext;


public:
	SoundHash();
	~SoundHash();
	
	void put(string name, Sound *s);
	Sound* get(string name); // will create sound with file name if not there
	bool remove(string name);  // will delete sound object
	int play(string name, int loop = 1);
	void SoundHash::stop(int id, bool immediate = false);
	bool alInit();
	void alClose();

};


#endif