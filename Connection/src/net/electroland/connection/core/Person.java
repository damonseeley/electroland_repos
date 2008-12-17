package net.electroland.connection.core;

/**
 * Adapted from CoopLights Person.java authored by Eitan Mendelowitz.
 * Revised by Aaron Siegel
 */

public class Person {
	public Integer id;
	public float x, y, z;						// position and height
	public float lastx, lasty, xvec, yvec;		// direction data
	public float[] pastvecs, tempvecs;			// used to average Y vector
	public float avgvec, sumvecs;				// constantly averaged Y vector
	public float[] pasthorpos, temphorpos;		// used to average X position (to prevent horizontal jitter)
	public float avghorpos, sumhorpos;
	public int lightx, lighty;					// location in lighting grid
	public int lastlightx, lastlighty;
	private float[] floatloc = new float[2];	// grid position as float
	private int[] intloc = new int[2];		// grid position quantized
	private int[] speakerloc = new int[2];
	public int element;
	public int gx, gy;							// grid dimensions
	public long birthdate;						// when they entered
	public float minlinkdist;					
	public boolean newLoc;
	public boolean newPerson;
	public boolean inside;
	public boolean paired;
	public int soundID;
	
	public Person(int idvalue, float xvalue, float yvalue, float zvalue, int gridx, int gridy) {
		id = new Integer(idvalue); // Integer object is used by hashmap so we might as well store id as an object
		x = xvalue;					// normalized x position
		y = yvalue;					// normalized y position
		z = zvalue;					// height
		lastx = x;					
		lasty = y;
		lastlightx = -100;
		lastlighty = -100;
		gx = gridx;					// x light count
		gy = gridy;					// y light count
		birthdate = System.currentTimeMillis();
		minlinkdist = 4;			// distance in light grid
		newPerson = true;
		inside = false;
		paired = false;
		element = -1;
		pastvecs = new float[Integer.parseInt(ConnectionMain.properties.get("vecAvgDuration"))];		// average long-ways vector over 2 seconds
		tempvecs = new float[Integer.parseInt(ConnectionMain.properties.get("vecAvgDuration"))];
		pasthorpos = new float[Integer.parseInt(ConnectionMain.properties.get("posAvgDuration"))];	// average horizontal position over a second
		temphorpos = new float[Integer.parseInt(ConnectionMain.properties.get("posAvgDuration"))];
	}
	
	public long getAge(){
		return System.currentTimeMillis() - birthdate;
	}
	
	public int getVec(){
		if(avgvec < 0){
			return 0;
		} else {
			return 1;
		}
		 
	}
	
	public float[] getFloatLoc(){
		if(x < 0){
			floatloc[0] = 0.001f;
		} else if(x > 1){
			floatloc[0] = 5.999f;
		} else {
			floatloc[0] = x*gx;
		}
		floatloc[1] = y*gy;
		return floatloc;
	}
	
	public int[] getIntLoc(){
		/*
		if(x < 0){
			intloc[0] = 0;
		} else if(x >= 1){
			intloc[0] = 5;
		} else {
			intloc[0] = (int)Math.floor(x*gx);
		}
		*/
		
		// this new code has horizontal position averaging to attempt to remove jitter
		if(avghorpos < 0){
			intloc[0] = 0;
		} else if(avghorpos >= 1){
			intloc[0] = 5;
		} else {
			intloc[0] = (int)Math.floor(avghorpos*gx);
		}
		intloc[1] = (int)Math.floor(y*gy);
		lightx = intloc[0];
		lighty = intloc[1];
		return intloc;
	}
	
	public int[] getForwardIntLoc(float compensation){
		if(x < 0){
			intloc[0] = 0;
		} else if(x >= 1){
			intloc[0] = 5;
		} else {
			intloc[0] = (int)Math.floor(x*gx);
		}
		intloc[1] = (int)Math.floor((y+(avgvec*compensation))*gy);
		lightx = intloc[0];
		lighty = intloc[1];
		return intloc;
	}
	
	public int[] getNearestSpeaker(){
		speakerloc[0] = (int)Math.floor(y*12) + 1;
		if(y < 0.5){
			speakerloc[1] = 2;
		} else {
			speakerloc[1] = 1;
		}
		return speakerloc;
	}
	
	public void setLoc(float xvalue, float yvalue){
		x = xvalue;
		y = yvalue;
		xvec = x - lastx;
		yvec = y - lasty;
		lastx = x;
		lasty = y;
		System.arraycopy(pastvecs, 0, tempvecs, 1, pastvecs.length-1);
		System.arraycopy(pasthorpos, 0, temphorpos, 1, pasthorpos.length-1);
		tempvecs[0] = yvec;
		temphorpos[0] = x;
		pastvecs = tempvecs;
		pasthorpos = temphorpos;
		sumvecs = 0;
		sumhorpos = 0;
		for(int i=0; i<pastvecs.length; i++){
			sumvecs += pastvecs[i];
		}
		for(int i=0; i<pasthorpos.length; i++){
			sumhorpos += pasthorpos[i];
		}
		avgvec = sumvecs/pastvecs.length;			// average long-ways vector
		avghorpos = sumhorpos/pasthorpos.length;	// average horizontal position
	}
}
