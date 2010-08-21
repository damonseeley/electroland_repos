#ifndef __RECORDER_H__
#define __RECORDER_H__

#include "TyzxCam.h"
#include "PersonTrackReceiver.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class CamRecorder {
public:
	DWORD curTime;
	DWORD stopTime;
	ofstream fileStream;
	int camCnt;
	TyzxCam** cams;

	CamRecorder(int camCnt, TyzxCam** cams) { this->camCnt = camCnt; this->cams = cams;}
	void setStopTime(DWORD stopTime) {  this->stopTime = stopTime;}

	void open(string s);
	void operator()() { loop(); }
	void loop();

};

class TrackRecorder {
public:
	DWORD curTime;
	DWORD stopTime;
	ofstream fileStream;
	PersonTrackReceiver* tracker;

	TrackRecorder(PersonTrackReceiver* tracker) { this->tracker = tracker;}
	void setStopTime(DWORD stopTime) {  this->stopTime = stopTime;}

	void open(string s);
	void operator()() { loop(); }
	void loop();

};

class Recorder {

public:
	boost::thread_group threadGroup;
	int camCnt;
	CamRecorder* camRecorders;
	TrackRecorder *tracker;
	string filename;

	Recorder(string filename, int camCnt, TyzxCam** cams, PersonTrackReceiver *tracker);
	void open();
	void run(DWORD duration);
	void close();
	~Recorder();

}
;

struct threadCallableTracker {
	TrackRecorder* obj;
	void operator()() { obj->loop(); }
};

struct threadCallableCam {
	CamRecorder* obj;
	void operator()() { obj->loop(); }
};

#endif