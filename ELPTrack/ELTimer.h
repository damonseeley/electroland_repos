#ifndef __EL_TIMER__
#define __EL_TIMER__

#define NOMINMAX
#include <Windows.h> // only on a Windows system
#undef NOMINMAX

class ELTimer {
public:
	__int64    TimeCounter;
	__int64    TimeFrequency;   
	float freqInv;
	float      lastTime;
	float	   curTime;
	float	nextEpoc;
	float	lastEpoc;
	int		frameCnt;
	bool       hirestimer;
	
	float requiredSleep;
	float fps;
	float actualFPS;

	ELTimer(float fps);
	~ELTimer();

	void sleep(float ms);
	float getTime();
	void maintainFPS();

};

#endif