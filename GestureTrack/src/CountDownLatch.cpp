/*#include "CountDownLatch.h"

CountDownLatch::CountDownLatch() {
	InitializeCriticalSection (&watingCritSect)
	releaseEvent = CreateEvent(0, FALSE, TRUE, 0);
}

void CountDownLatch::addUser() {
	users++;
}

void CoundDownLatch::release() {
	waiting = users;
	SetEvent(releaseEvent);
}

void CountDownLatch::wait() {
	bool needToWait = false;
	EnterCriticalSection(&watingCritSect);
	waiting--;
	if(waiting ==0) {
		needToWait = true;
	}
	LeaveCriticalSection(&watingCritSect);
	if(needToWait) {
		// wait for event
	} else {
		release();
	}
}

CountDownLatch::~CountDownLatch() {
	release();
	DeleteCriticalSection  (&watingCritSect)
	CloseHandle(releaseEvent);
}
*/