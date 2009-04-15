package net.electroland.laface.tracking;

/**
 * A Mover is an artifact of the direction, velocity, and acceleration/deceleration of
 * a Candidate that was being tracked. The Mover is an abstraction layer to combat noise.
 * 
 * @author Aaron Siegel
 */

public class Mover {
	
	private float x, y;			// current position (normalized)
	private long startTime;
	
	public Mover(Candidate successor){
		startTime = System.currentTimeMillis();
	}
	
	public float getX(){
		// TODO calculate based on time
		return x;
	}
	
	public float getY(){
		// TODO calculate based on time
		return y;
	}

}
