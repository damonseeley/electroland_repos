#include "soundsourcepool.h"
#include "Sound.h"
#include "Profile.h"

SoundSourcePool::SoundSourcePool()
{
	NUM_SOURCES = CProfile::theProfile->Int("MaxVoices", 48);
	source = new ALuint[NUM_SOURCES];
	buffer = new ALuint[NUM_SOURCES];
	loopCnt = new int[NUM_SOURCES];

}

SoundSourcePool::~SoundSourcePool()
{
	alDeleteBuffers(NUM_SOURCES, buffer);
	alDeleteSources(NUM_SOURCES, source);
	delete[] source;
	delete[] buffer;
	delete[] loopCnt;

}


void SoundSourcePool::genSources() {
	curSrc = 0;
	alGenSources(NUM_SOURCES, source);
	ALint error = alGetError(); 
    if(error != AL_NO_ERROR) 
    {
		timeStamp(); clog << "ERROR  Failed to create sources will not play sounds " << alGetString(error) << "\n";
	}

	alGenBuffers(NUM_SOURCES, buffer);
	error = alGetError(); 
    if(error != AL_NO_ERROR) 
    {
		timeStamp(); clog << "ERROR  Failed to create buffers will not play sounds " << alGetString(error) << "\n";
	}

}
void SoundSourcePool::unqueueSource(int src, int size) {
		ALuint tmp;
		alSourceUnqueueBuffers(source[src], size, &tmp);
	}

bool SoundSourcePool::makeFree(int i) { // and free
	int state;
	alGetSourcei(source[i], AL_SOURCE_STATE, &state);
	if ((state == AL_STOPPED) || (state == AL_INITIAL)) {
				alDeleteSources(1, &source[i]);
				alGenSources(1, &source[i]);
		return true;
	}
	int loops = loopCnt[i];
	if ( (loops > 0) && (state == AL_PLAYING) ) {
		alGetSourcei(source[i], AL_BUFFERS_PROCESSED, &state);
		if (state == loops) {
			unqueueSource(i, loops);
				alDeleteSources(1, &source[i]);
				alGenSources(1, &source[i]);
			return true;
		}
	}
	return false;
}
int SoundSourcePool::getFreeSource() {
	int i = curSrc;
	bool foundFree = false;
	while( (! foundFree) && (i < NUM_SOURCES)) 
		foundFree = makeFree(i++);

	if (foundFree) {
		curSrc = i;
		return i - 1;
	}

	i =0;
	while( (! foundFree) && (i < curSrc)) 
		foundFree = makeFree(i++);

	if (foundFree) {
		curSrc = i;
		return i - 1;
	}

	return -1;

}
void SoundSourcePool::loadBuffer(Sound *snd, int bufferId) {
//		alDeleteBuffers(1, &buffer[bufferId]);
//		alGenBuffers(1, &buffer[bufferId]);

		// shouldn't have to do this but buffers are getting currupted and repeating
		alBufferData(buffer[bufferId],snd->format,snd->data,snd->size,snd->freq);
		ALint error = alGetError(); 
		if(error != AL_NO_ERROR)  {
			timeStamp(); clog << "WARNING  Unable to buffer sound due to " << alGetString(error) << "\n";
		}
}

