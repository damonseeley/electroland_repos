package net.electroland.laface.tracking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

/**
 * A Mover is an artifact of the direction, velocity, and acceleration/deceleration of
 * a Candidate that was being tracked. The Mover is an abstraction layer to combat noise.
 * 
 * @author Aaron Siegel
 */

public class Mover {
	
	private Candidate successor;
	private int id;
	private float x, y;			// current position (normalized)
	private float startx, starty;
	private long startTime;
	private long xduration, yduration, minduration, maxduration;
	private float xvec, yvec;
	private List<MoverListener> listeners;
	private boolean dead;
	private boolean trackAlive = false;
	
	public Mover(Candidate successor){
		this.successor = successor;
		id = successor.getID();
		Vector<Float> movement = successor.getSpeed();
		if(movement == null){
			dead = true;
		} else {
			xvec = movement.get(0);
			yvec = movement.get(1);
			if(xvec < 0){
				x = 1;
			} else {
				x = 0;
			}
			y = successor.y;
			startx = x;
			starty = y;
			xduration = (long)movement.get(2).longValue();
			yduration = (long)movement.get(3).longValue();
			//System.out.println("xduration: "+xduration);
			if(xduration < yduration){
				minduration = xduration;
				maxduration = yduration;
			} else {
				minduration = yduration;
				maxduration = xduration;
			}
			dead = false;
		}
		listeners = new ArrayList<MoverListener>();
		startTime = System.currentTimeMillis();
	}
	
	public float getX(){
		checkIfDead();
		//System.out.println((System.currentTimeMillis() - startTime)+" "+xduration+" "+(float)(System.currentTimeMillis() - startTime)/xduration);
		if(trackAlive){
			x = successor.x;
		} else {
			if(xvec >= 0){
				x = (((float)(System.currentTimeMillis() - startTime)/xduration) * (1-startx)) + startx;	// return normalized X position
			} else {
				x = startx - (((float)(System.currentTimeMillis() - startTime)/xduration) * startx);		// return normalized X position
			}
		}
		return x;
	}
	
	public float getXvec(){
		return xvec;
	}
	
	public float getYvec(){
		return yvec;
	}
	
	public float getY(){
		checkIfDead();
		if(trackAlive){
			y = successor.y;
		} else {
			if(yvec >= 0){
				y = (((float)(System.currentTimeMillis() - startTime)/yduration) * (1-starty)) + starty;	// return normalized X position
			} else {
				y = starty - (((float)(System.currentTimeMillis() - startTime)/yduration) * starty);		// return normalized X position
			}
		}
		return y;
	}
	
	public int getID(){
		return id;
	}
	
	private void checkIfDead(){
		if((xvec < 0 && x < -0.1) || (xvec > 0 && x > 1.1)){			// if outside X constraints + margin...
			dead = true;
			Iterator<MoverListener> iter = listeners.iterator();
			while(iter.hasNext()){
				MoverListener ml = iter.next();
				ml.moverEvent(this);									// notify listeners of death
			}
		}
	}
	
	public boolean isDead(){
		if(dead){
			return true;
		}
		return false;
	}
	
	public void trackDied(){
		trackAlive = false;
	}
	
	public void addListener(MoverListener l){
		listeners.add(l);
	}
	
	public long getMinDuration(){
		return minduration;
	}
	
	public long getMaxDuration(){
		return maxduration;
	}

}
