package net.electroland.connection.core;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Iterator;

import net.electroland.udpUtils.UDPParser;

/**
 * Adapted from CoopLights PersonTracker.java authored by Eitan Mendelowitz.
 * Revised by Aaron Siegel
 */

public class PersonTracker extends UDPParser {
	
	public ConcurrentHashMap<Integer, Person> people = new ConcurrentHashMap<Integer, Person>();
	public float xscale, yscale;
	public float xoffset, yoffset;
	public boolean xflip, yflip, swap;
	public float skew;
	public boolean trackingIndividual;
	public int individual;
	public int exitcounter;
	public float exitavg;
	
	public PersonTracker(int port) throws SocketException, UnknownHostException {
		super(port);
		System.out.println("Connection opened on port "+ port);
		exitcounter = 0;
		trackingIndividual = false;
		xscale = 1/(float)Integer.parseInt(ConnectionMain.properties.get("gridWidth"));
		yscale = 1/(float)Integer.parseInt(ConnectionMain.properties.get("gridHeight"));
		yoffset = Integer.parseInt(ConnectionMain.properties.get("xoffset")) / (float)Integer.parseInt(ConnectionMain.properties.get("gridWidth"));
		xoffset = Integer.parseInt(ConnectionMain.properties.get("yoffset")) / (float)Integer.parseInt(ConnectionMain.properties.get("gridHeight"));
		xflip = Boolean.parseBoolean(ConnectionMain.properties.get("flipX"));
		yflip = Boolean.parseBoolean(ConnectionMain.properties.get("flipY"));
		swap = Boolean.parseBoolean(ConnectionMain.properties.get("swapXY"));
		skew = Float.parseFloat(ConnectionMain.properties.get("skew"));
	}
	
	public void handleTrackInfo(int id, int x, int y, int h) {
		Integer idObj = new Integer(id);					// convert to Integer object
		Person p = people.get(idObj);						// attempt to retrieve Person object
		if(!trackingIndividual){
			individual = id;
			trackingIndividual = true;
		} else {
			if(people.containsKey(individual)){
			}
		}
		float newx = x*xscale;
		float newy = y*yscale;
		skew = Float.parseFloat(ConnectionMain.properties.get("skew"));	// still in centimeters!
		
		if(swap){
			float tempx = newx;
			newx = newy;
			newy = tempx;
		}
		if(xflip){
			newx = 1 - newx;
		}
		if(yflip){
			newy = 1 - newy;
		}
		
		newx += skew*yscale*newy;	// skew * normalized x position = derivative function
		
		if(p == null) {									// if it doesn't exist...
			p = new Person(idObj, newx + xoffset, newy + yoffset, h, 6, 28);	// create a new person
			people.put(idObj, p);							// add person to people table
		} else {											// if it does exist...
			p.setLoc(newx + xoffset, newy + yoffset);		// update it's location
			p.z = h;
		}
	}
	
	public void handleExit(int id) {						// remove person from hash table
		if(id == individual){
			individual = 0;
			trackingIndividual = false;
		}
		people.remove(id);
		exitcounter++;
	}
	
	public int peopleCount(){
		return people.size();
	}
	
	public int[] getKeys(){
		Integer[] keylist = new Integer[people.size()];		// make array of person ID#'s
		int[] keys = new int[people.size()];
		people.keySet().toArray(keylist);
		for(int i=0; i<keylist.length; i++){
			keys[i] = keylist[i].intValue();
		}
		return keys;
	}
	
	public Iterator<Person> getPersonIterator(){
		return people.values().iterator();
	}
	
	public Person getPerson(int id){
		return people.get(new Integer(id));
	}
}
