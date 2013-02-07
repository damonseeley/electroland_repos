#ifndef __SOUNDSOURCEPOOL__
#define __SOUNDSOURCEPOOL__

#include"al.h"
#include"alc.h"
#include"alut.h"

#include "Globals.h"
#include "Sound.h"


class SoundSourcePool
{
public:
	int curSrc;
	int NUM_SOURCES;
	ALuint *source;
	ALuint  *buffer;
	int *loopCnt;


public:
	SoundSourcePool();
	~SoundSourcePool();
	void genSources();
	int getFreeSource();
	bool makeFree(int i); // if possible 
	void unqueueSource(int src, int size);
	void loadBuffer(Sound *snd, int bid);

};

#endif

