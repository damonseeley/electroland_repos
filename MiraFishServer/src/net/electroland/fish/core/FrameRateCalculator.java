package net.electroland.fish.core;

public class FrameRateCalculator {
	public  float fps; //average
	public float elapsedTime; // average elapsed time in ms per frame
	public long instElapseTime;
	public long curTime;
	public float standardDeviation = 0;
	public float minVar = 0;
	public float maxVar = 0;

	
	private long[] timeCache;
	private int timeCacheSlot = 0;
	private boolean isReady = false;

	
	
	public FrameRateCalculator(int cacheSize, float anticipatedFSP) {
		timeCache = new long[cacheSize];
		fps = anticipatedFSP;
		elapsedTime = 1000.0f / fps;
	}
	public void frame(long sysTime) {
		long lastTime = curTime;

		curTime = sysTime;
		instElapseTime = curTime - lastTime;
		long pastTime = timeCache[timeCacheSlot];
		timeCache[timeCacheSlot++] = curTime;
		
		if(isReady) {
			long timeDiff = curTime - pastTime;
			elapsedTime = (float) timeDiff / (float) timeCache.length;
			fps = (1000.0f * (float)timeCache.length) /(float) timeDiff;
			timeCacheSlot = (timeCacheSlot >= timeCache.length) ? 0 : timeCacheSlot;
			
			float dev = (curTime - lastTime) - elapsedTime;
			
			minVar = (dev < minVar) ? dev : minVar;
			maxVar = (dev > maxVar) ? dev : maxVar;
			
			dev *= dev;
			
			
			dev *= .01;
			float newDev = standardDeviation;
			newDev *= .99;
			newDev += dev;
			standardDeviation = dev;
			
		} else {
			if(timeCacheSlot >= timeCache.length) {
				timeCacheSlot = 0;
				isReady = true;
			}
			
		}
	}
		
	
	public void frame() {
		frame(System.currentTimeMillis());
	}

}
