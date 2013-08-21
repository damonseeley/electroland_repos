#include "ELTimer.h"

// I've coded this 5 differnt ways.
// the true fps is consistantly about 1.8 times the desired fps (in release) and 1.3 times the diresred fps in debug.
// wierd!
ELTimer::ELTimer(float fps) {

	QueryPerformanceFrequency((LARGE_INTEGER *)&TimeFrequency);
	freqInv = 1.0f/(float)TimeFrequency;
	if (!QueryPerformanceCounter((LARGE_INTEGER *)&TimeCounter))		
		 hirestimer= false;		   
	else hirestimer= true;	
	
	this->fps = fps;
	requiredSleep=  (1000.0f/fps);
	curTime = getTime();
	lastTime= curTime;
	frameCnt = 0; 
	lastEpoc = curTime;
	nextEpoc = curTime + 1000.0f;
	actualFPS = fps;
}

void ELTimer::sleep(float ms) 
{ 
	
    static HANDLE Timer = CreateWaitableTimer( NULL, FALSE, NULL ); // Determine time to wait.  
    LARGE_INTEGER WaitTime; WaitTime.QuadPart = (LONGLONG)(ms * -10000); 
    if (WaitTime.QuadPart >= 0) 
        return; // Give up the rest of the frame.  

    if (!SetWaitableTimer(Timer, &WaitTime, 0, NULL, NULL, FALSE)) 
        return; DWORD Result = MsgWaitForMultipleObjects(1, &Timer, FALSE, INFINITE, QS_ALLINPUT); 
		

} 


float ELTimer::getTime()
{	
	if (hirestimer)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)&TimeCounter);
		return (1000.0f * (float)TimeCounter * freqInv); //convert to ms
	} else 
		return (float) timeGetTime();					
}

void ELTimer::maintainFPS() // in frames per second
{
	lastTime = curTime;
	curTime = getTime();
	if(curTime >= nextEpoc && frameCnt) {
		actualFPS = (float) 1000 * frameCnt/ (float)(curTime-lastEpoc);
		frameCnt = 0;
		lastEpoc = curTime;
		nextEpoc = curTime + 1000;
	} else {
		frameCnt++;
	}
	float sleepUntil = lastTime + requiredSleep;

	float realCurTime = getTime();
	while(realCurTime < sleepUntil) {
		float dt = realCurTime-lastTime;
		if(dt < requiredSleep) {
			sleep(requiredSleep - dt);
		}
		realCurTime = getTime();
	}
}