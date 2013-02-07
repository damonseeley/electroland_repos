#include "soundhash.h"
#include "profile.h"

SoundHash *SoundHash::theSoundHash = NULL;

SoundHash::SoundHash() {
	if (theSoundHash == NULL) {
		theSoundHash = this;
		useSound = CProfile::theProfile->Bool("audioOn", true);
		if(! useSound) return;
		if(AL_FALSE == alInit()) {
			useSound = false;
			return;
		}
		alGetError();
		alcGetError( g_currentDevice );


	}
}

ALCdevice *SoundHash::g_currentDevice= NULL;
ALCcontext *SoundHash::g_currentContext = NULL;

bool SoundHash::alInit() {

	//ALchar *aString;
	g_currentDevice = alcOpenDevice(NULL);
	if(NULL == g_currentDevice)
	{
		timeStamp(); clog << "WARNING  Unable to open default sound device\n";
		return AL_FALSE;
	}
	alcGetError( g_currentDevice );
	g_currentContext = alcCreateContext(g_currentDevice, NULL);
	if(NULL == g_currentContext)
	{
		timeStamp(); clog << "WARNING  Unable to get current sound context\n";
		return AL_FALSE;
	}
	alcMakeContextCurrent(g_currentContext);

	ssp.genSources();

	return AL_TRUE;
}

void SoundHash::alClose()
{
    alcDestroyContext(g_currentContext);
    alcCloseDevice(g_currentDevice);
}
SoundHash::~SoundHash() {
	if(! useSound) return;
	hash_map<string,Sound*>::iterator curr, end;
	end = sounds.end();
    for(curr = sounds.begin(); curr != end; curr++){
		delete (curr->second);
    }
		alClose();

}

void SoundHash::put(string name, Sound *s) {	
	sounds.insert(entry(name, s));
}

bool SoundHash::remove(string name) {
	Sound* s = get(name);
	if (s != NULL) {
		delete s;
		sounds.erase(name);
		return true;
	}
	return false;


}
// will loud sound if needed
Sound* SoundHash::get(string name) {

	hash_map<string,Sound*>::iterator it;
	it = sounds.find(name);
  if(it != sounds.end())
   return it->second;
  else {
//		ALint	    error;
	Sound *s = new Sound();
	alutLoadWAVFile(("wavs/" + name).c_str(),&s->format,&s->data,&s->size,&s->freq,&s->loop );
	if( s->data == NULL)
	{
		timeStamp(); clog << "WARNING  Failed to load sound " << name << "\n";
		delete s;
		return NULL;
	} else {
//		s->loadBuffer();
		put(name, s);
	}
	return s;
  }

}
void SoundHash::stop(int id, bool immediate) {
	if(! useSound) return;
	if(immediate) {
		alSourceStop(ssp.source[id]);
	} else {
		alSourcei(ssp.source[id], AL_LOOPING, AL_FALSE);
	}
		ssp.loopCnt[id] = 1;

}

int SoundHash::play(string name, int loop) {
	if(! useSound) return -1;
	Sound *snd = get(name);
	if(snd == NULL) return -1;
	
	int srcId = ssp.getFreeSource();
	if(srcId < 0) return -1;
//	cout << "playing on souce id " << srcId << endl;

	ssp.loadBuffer(snd, srcId);
	/*
	alSourcei(ssp.source[srcId], AL_BUFFER, ssp.buffer[srcId]);  {
	ALint error = alGetError(); 
		if(error != AL_NO_ERROR)  {
			timeStamp(); clog << "WARNING  Unable to queue sound due to " << alGetString(error) << "\n";
		}	
	}
	*/
	if(loop < 1) { // loop forever
		    alSourcei(ssp.source[srcId], AL_LOOPING, AL_TRUE);
			alSourceQueueBuffers(ssp.source[srcId], 1, &ssp.buffer[srcId]);
			ALint error = alGetError() ;
			if(AL_NO_ERROR != error) {
				timeStamp(); clog << "WARNING  Unable to queue sound " << name <<  " due to " << alGetString(error) << "\n";
				return -1;
			}
	} else { //loop a fixed number of times using queue
		int cnt = 0;
		alSourcei(ssp.source[srcId], AL_LOOPING, AL_FALSE);
		while(cnt < loop) {
			alSourceQueueBuffers(ssp.source[srcId], 1, &(ssp.buffer[srcId]));
			ALint error = alGetError() ;
			if(AL_NO_ERROR != error) {
				timeStamp(); clog << "WARNING  Unable to queue sound " << name <<  " due to " << alGetString(error) << "\n";
				return -1;
			}
			cnt++;
		}
	}

	ssp.loopCnt[srcId] = loop;
	 alSourcePlay( ssp.source[srcId] );
			ALint error = alGetError() ;
			if(AL_NO_ERROR != error) {
				timeStamp(); clog << "WARNING  Unable to play sound " << name <<  " due to " << alGetString(error) << "\n";
				return -1;
			}

	return srcId;

}



 
