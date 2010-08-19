#include "SyncedThreadLoop.h"

//void SyncedThreadLoop::threadfunc(SyncedThreadLoop *loop) {
//	loop->run();
//}

SyncedThreadLoop::SyncedThreadLoop(boost::barrier* bar){
	this->bar = bar;
}

void SyncedThreadLoop::start() {
	isRunning = true;
	threadCallable c;
	c.obj = this;
	thread = new boost::thread(c);
//	thread->join();
}

void SyncedThreadLoop::stop() {
	isRunning = false;
	thread->join();
}


void SyncedThreadLoop::loop() {
	init();
	run();
	while(isRunning) {		
		bar->wait();
		run();
	}
}
SyncedThreadLoop::~SyncedThreadLoop() {
	stop();
	delete thread;
}