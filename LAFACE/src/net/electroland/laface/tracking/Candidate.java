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
	
	public float x, y;
	private Vector<Vector<Float>> locations;
	private Vector<Long> times;

	public Candidate(){
		locations = new Vector<Vector<Float>>();
		times = new Vector<Long>();
	}
	
	public void addLocation(float x, float y){
		Vector<Float> loc = new Vector<Float>();
		loc.add(x);
		loc.add(y);
		locations.add(loc);
		times.add(System.currentTimeMillis());
	}
	
	public Vector<Vector<Float>> getLocations(){
		return locations;
	}
	
}
