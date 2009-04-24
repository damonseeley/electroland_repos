package net.electroland.laface.tracking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobTracker.util.ElProps;

/**
 * A Target monitors a Track instance until it dies or holds its position,
 * at which point the Target will continue along the same vector at the same rate.
 * 
 * @author Aaron Siegel
 */

public class Target {
	
	private Track track;
	private int id;
	private float x, y;				// current position (normalized)
	private float pastx, pasty;		// previous position
	private float width, height;		// width and height
	private float startx, starty;
	private float lasttrackx, lasttracky;
	private LinkedList<Float> xpositions;
	private LinkedList<Float> ypositions;
	private int sampleCount;
	private long startTime;
	private float xvec, yvec;
	private boolean trackAlive;
	private boolean dead;
	private List<TargetListener> listeners;	// TODO should be changed to TargetListener

	public Target(Track track){
		this.track = track;
		id = track.id;
		width = track.width / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
		height = track.height / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcHeight").toString());
		startx = pastx = x = track.x / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
		starty = pasty = y = track.y / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcHeight").toString());
		trackAlive = true;
		dead = false;
		sampleCount = 5;
		xpositions = new LinkedList<Float>();
		ypositions = new LinkedList<Float>();
		listeners = new ArrayList<TargetListener>();
		startTime = System.currentTimeMillis();
	}
	
	public int getID(){
		return id;
	}
	
	public float getX(){
		checkIfDead();
		if(trackAlive){
			float newx = track.x  / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
			lasttrackx = newx;				// most recent track location stored
			if(x == newx){					// if most recent track location is current location...
				trackAlive = false;			// track must be dead, stuck, or stopped
			} else {						// track still alive...
				pastx = x;					// store last X position
				if(xpositions.size() > sampleCount){	// prevent queue from getting too long
					xpositions.getFirst();	// pop off the oldest location
				}
				xpositions.addLast(pastx);	// append the newest location
				x = newx;
			}			
		} else {							// if track NOT alive
			float newx = track.x  / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
			if(lasttrackx != newx){			// check track location to see if it's still dead
				trackAlive = true;			// if not, set alive again
			} else {						// if still dead...
				//float xdiff = x - pastx;	// TODO change this to an average speed based on multiple past points
				Iterator<Float> iter = xpositions.iterator();
				float lastpos = 0;
				if(iter.hasNext()){			// grab first one
					lastpos = iter.next();
				}
				float totaldiff = 0;
				while(iter.hasNext()){
					Float xpos = iter.next();
					totaldiff += xpos - lastpos;
					lastpos = xpos;
				}
				float xdiff = totaldiff/xpositions.size();
				pastx = x;
				x += xdiff;
			}
		}
		return x;
	}
	
	public float getY(){
		checkIfDead();
		if(trackAlive){
			float newy = track.y  / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcHeight").toString());
			if(y == newy){
				//trackAlive = false;
			} else {
				pasty = y;
				y = newy;
			}
		} else {
			float ydiff = y - pasty;
			pasty = y;
			y += ydiff;
		}
		return y;
	}
	
	public float getWidth(){
		if(trackAlive){
			width = track.width / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
		}
		return width;
	}
	
	public float getHeight(){
		if(trackAlive){
			height = track.height / (float)Integer.parseInt(ElProps.THE_PROPS.get("srcHeight").toString());
		}
		return height;
	}
	
	public float getXvec(){
		return x - pastx;
	}
	
	public void trackDied(){
		//System.out.println("track "+id+" died");
		trackAlive = false;
		//die();
	}
	
	public boolean isDead(){
		return dead;
	}
	
	public void addListener(TargetListener l){
		listeners.add(l);
	}
	
	private void checkIfDead(){
		//System.out.println(id+" "+x);
		if((getXvec() > 0 && x > 1) || (getXvec() < 0 && x < 0)){	// wait till they move off screen to die
			dead = true;
			die();
		}
	}
	
	private void die(){
		Iterator<TargetListener> iter = listeners.iterator();
		while(iter.hasNext()){
			TargetListener tl = iter.next();
			tl.targetEvent(this);									// notify listeners of death
		}
	}
	
}
