/*
#ifndef __COUNT_DOWN_LATCH__
#define __COUNT_DOWN_LATCH__
class CountDownLatch {
public: 
	CountDownLatch();
	void addUser();
	void wait();
	void release();
	virtual ~CountDownLatch();
private:
	HANDLE releaseEvent;
	CRITICAL_SECTION watingCritSect;
	int users;
	int waiting;
}
#endif*/