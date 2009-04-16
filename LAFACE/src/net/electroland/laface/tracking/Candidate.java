package net.electroland.laface.tracking;

import java.util.Vector;

/**
 * A Candidate is associated with a Track, and is used to retain a history of the
 * tracks location in order to assess the velocity and direction of the Track, as
 * well as whether it's accelerating or decelerating.
 * 
 * If a Candidate survives the number of required samples and maintains a velocity
 * above the minimum, a new Mover object will be created with the Candidates properties
 * and the Candidate will be destroyed.
 * 
 * @author Aaron Siegel
 */

public class Candidate {
	
	private int id;
	public float x, y;							// current location (normalized)
	private Vector<Vector<Float>> locations;	// past locations (normalized)
	private Vector<Long> times;				// sample times (milliseconds)

	public Candidate(int id){
		this.id = id;
		locations = new Vector<Vector<Float>>();
		times = new Vector<Long>();
	}
	
	public void addLocation(float x, float y){
		this.x = x;
		this.y = y;
		Vector<Float> loc = new Vector<Float>();
		loc.add(this.x);
		loc.add(this.y);
		locations.add(loc);
		times.add(System.currentTimeMillis());
	}
	
	public boolean isStatic(){
		float xdiff = locations.get(locations.size()-1).get(0) - locations.get(0).get(0);	// distance between first and last sample.
		float ydiff = locations.get(locations.size()-1).get(1) - locations.get(0).get(1);
		if(xdiff == 0 && ydiff == 0){
			return true;
		}
		return false;
	}
	
	// this function returns [0] X unit vector, [1] Y unit vector, [2] X duration to edge, [3] Y duration to edge
	public Vector<Float> getSpeed(){
		// TODO evaluate all sample points for acceleration/deceleration trend and rate
		long samplediff = System.currentTimeMillis() - times.get(0);						// time in between first and last sample.
		float xdiff = locations.get(locations.size()-1).get(0) - locations.get(0).get(0);	// distance between first and last sample.
		float ydiff = locations.get(locations.size()-1).get(1) - locations.get(0).get(1);
		//System.out.println(xdiff);
		if(xdiff == 0){
			return null;
		}
		float hypodiff = (float)Math.sqrt(xdiff*xdiff + ydiff*ydiff);
		
		// TODO check how long it will take before the mover is outside the raster at the current speed on this vector
		float xscale, yscale;
		if(xdiff >= 0){
			xscale = (1-Math.abs(locations.get(locations.size()-1).get(0)))/xdiff;		// multiple of how much longer it will take to 
		} else {																		// complete the remaining distance at this speed.
			xscale = Math.abs(locations.get(locations.size()-1).get(0)/xdiff);
			//System.out.println("xduration: "+ (samplediff * xscale) +" "+xscale);
			//System.out.println(Math.abs(locations.get(locations.size()-1).get(0)) +" "+ xdiff);
		}
		if(ydiff >= 0){
			yscale = (1-Math.abs(locations.get(locations.size()-1).get(1)))/ydiff;		
		} else {
			yscale = Math.abs(locations.get(locations.size()-1).get(1)/ydiff);
		}
		
		Vector<Float> moverVec = new Vector<Float>();
		moverVec.add(xdiff/hypodiff);					// unit vector
		moverVec.add(ydiff/hypodiff);	
		moverVec.add(samplediff * xscale);				// time between first and last sample multiplied by the ratio 
		moverVec.add(samplediff * yscale);				// of remaining distance to what was covered between sample points.
		return moverVec;
	}
	
	public Vector<Vector<Float>> getLocations(){
		return locations;
	}
	
	public int getID(){
		return id;
	}
	
}
