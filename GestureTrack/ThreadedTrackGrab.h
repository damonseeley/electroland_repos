#ifndef __THREADED_TRACK_GRAB_H__
#define __THREADED_TRACK_GRAB_H__

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "SyncedThreadLoop.h"
#include "PersonTrackReceiver.h"
#include "UDPSender.h"
#include "TrackHash.h"


class TrackGrabber;
class TrackGrabSender;


class ThreadedTrackGrab {
public:
	TrackGrabber *grabber;
	TrackGrabSender  *sender;

	boost::barrier *grabBarrier;
	boost::barrier *sendBarrier;
	

	TrackHash *grabbedTrackHash;
	TrackHash *curTrackHash;
	TrackHash *sendingTrackHash;

	PersonTrackReceiver *receiver;
	UDPSender *udpsender;


	void start();
	void grab();
	void send();

	ThreadedTrackGrab(PersonTrackReceiver *receiver, UDPSender *sender);

	TrackHash* getCurrentHash(); // swap grab with cur return cur
	void trackHashUpdated(); // swap cur with send send cur

};


class TrackGrabber : public SyncedThreadLoop {
public:
	ThreadedTrackGrab* track;

	TrackGrabber(boost::barrier *bar, ThreadedTrackGrab *track) :  SyncedThreadLoop(bar) {
		this->track = track;
	}
	
	virtual void run() {
		track->grab();
	}
}

;

class TrackGrabSender : public SyncedThreadLoop {
public:
	ThreadedTrackGrab* track;

	TrackGrabSender(boost::barrier *bar, ThreadedTrackGrab *track) :  SyncedThreadLoop(bar) {
		this->track = track;
	}
	
	virtual void run() {
		track->send();
	}
}

;

#endif