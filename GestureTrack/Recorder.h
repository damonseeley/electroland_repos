#ifndef __RECORDER_H__
#define __RECORDER_H__

#include "TyzxCam.h"
#include "PersonTrackReceiver.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class Recorder {

public:
	int camCnt;
	TyzxCam** cams;
	PersonTrackReceiver *tracker;
	string filename;

	ofstream trackFile;
	ofstream *zimgFile;

	Recorder(string filename, int camCnt, TyzxCam** cams, PersonTrackReceiver *tracker);
	void open();
	void run(DWORD duration);
	void close();
	~Recorder();

}

;
#endif