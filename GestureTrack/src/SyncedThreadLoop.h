#ifndef __SYNCED_THREAD_LOOP_H__
#define __SYNCED_THREAD_LOOP_H__


#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "Windows.h"




class SyncedThreadLoop {
public:
	
	
	bool isRunning;
	boost::barrier* bar;
	boost::thread* thread;


	SyncedThreadLoop(boost::barrier* bar);
	void start();
	void stop();

//	static void threadfunc(SyncedThreadLoop *loop);

	void loop();
	virtual void init() { };
	virtual void run() { };
	virtual ~SyncedThreadLoop();
	
};

struct threadCallable {
	SyncedThreadLoop* obj;
	void operator()() { obj->loop(); }
};


#endif
