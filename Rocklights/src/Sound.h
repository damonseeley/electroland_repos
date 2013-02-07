#ifndef __SOUND_H__
#define __SOUND_H__

#include"al.h"
#include"alc.h"
#include"alut.h"

#include "Globals.h"


class Sound {
public:
	ALsizei     size, freq;
	ALenum	    format;
	ALvoid	    *data;
	ALboolean   loop;

//	ALuint  buffer;

public:
	~Sound();
}
;
#endif
