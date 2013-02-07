#include "Sound.h"
/*
void Sound::loadBuffer(ALuint &buffer) {
		// Copy data into ALBuffer

	ALint error = alGetError();
	if(error != AL_NO_ERROR) {
		timeStamp(); clog << "WARNING  Failed to create buffer"  << alGetString(error) << "\n";

		return;
	}
//	alBufferData( BufferID, format, data, size, freq );

	alBufferData(buffer,format,data,size,freq);
}

void Sound::free(ALuint &buffer) {
	alDeleteBuffers(1, &buffer);
}
*/
Sound::~Sound() {
	alutUnloadWAV(format,data,size,freq);
}