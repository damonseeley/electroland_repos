#ifndef __THREADED_TRACK_SENDER_H__
#define __THREADED_TRACK_SENDER_H__

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "SyncedThreadLoop.h"
#include "UDPSender.h"
#include "TrackHash.h"


class TrackSender;


class ThreadedTrackSender {
public:
	TrackSender  *sender;

	boost::barrier *sendBarrier;
	

	TrackHash *curTrackHash;
	TrackHash *sendingTrackHash;

	UDPSender *udpsender;


	void start();
	void send();

	ThreadedTrackSender(UDPSender *sender);

	void setCurrentHash(TrackHash* hash);

};



class TrackSender : public SyncedThreadLoop {
public:

	ThreadedTrackSender* trackSender;
	TrackSender(boost::barrier *bar, ThreadedTrackSender *trackSender) :  SyncedThreadLoop(bar) {
		this->trackSender = trackSender;
	}
	
	virtual void run() {
		trackSender->send();
	}
}

;

#endif