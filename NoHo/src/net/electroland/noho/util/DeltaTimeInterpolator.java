package net.electroland.noho.util;

/**
 * interpolates from 1 -> 0
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class DeltaTimeInterpolator {
	public double denominator;
	public double timeLeft;
	public long origTime;

	public DeltaTimeInterpolator(long time) { 
		reset(time);
	}
	
	/**
	 * 
	 * @param dt - increment the interpolator by dt
	 * @return
	 */
	public double interp(long dt) {
		double d =  timeLeft * denominator;
		timeLeft -= dt;
		return d;
	}
	
	/**
	 * is done interpolating
	 * @return
	 */
	public boolean isDone() {
		return timeLeft < 0;
	}
	
	/**
	 * @param time - time overwhich to start interpolating  
	 */
	public void reset(long time) {
		origTime = time;
		if(time == 0) {
			timeLeft = 0;
			denominator = 0;
		} else {
			denominator = 1.0 / (double) time;
			timeLeft = time;
		}
	}
	public void reset() {
		reset(origTime);
	}
	
}
