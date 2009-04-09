package net.electroland.noho.util;

public class SimpleTimer {

	private long startTime;
	private long timeOut;

	public SimpleTimer(long timeOut){
		this.startTime = System.currentTimeMillis();
		this.timeOut = timeOut;
	}

	public boolean isDone() {
		if (System.currentTimeMillis() - startTime > timeOut){
			return true;
		} else
			return false;
	}
	
	public double percentDone() {
		double percentDone = (double)(System.currentTimeMillis() - startTime)/(double)timeOut;
		//System.out.println(percentDone);
		return percentDone;
	}
	
	public void resetTimer() {
		this.startTime = System.currentTimeMillis();
	}
	
	
	
	
}





