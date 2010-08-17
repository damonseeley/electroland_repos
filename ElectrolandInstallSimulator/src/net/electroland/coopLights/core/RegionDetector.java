package net.electroland.coopLights.core;

import java.awt.Graphics;
import java.util.Vector;

import net.electroland.installsim.core.Person;

/***
 * Use RegionDetector's for implementing "trip wrires"
 * @author eitan
 *
 */
public class RegionDetector {
	Vector<Region> regions = new Vector<Region> ();
	
	
	public  RegionDetector() {		
	}
	
	
	
	/**
	 * 	 
	 * @param person
	 * @return Returns the id for the first region that contains person.  Else returns -1.
	 */public int getRegion(Person person) {
		// trying to use this new 5.0 syntax more for loop
		for (Region r  : regions) {
			if(r.contains(person.x, person.y)) {
				return r.id;
			}
		}
		return -1;
	}
	
	public void addRegion(int id, float top, float left,float bot, float right){
		regions.add(new Region(id, top, left, bot, right));
	}
	
	public boolean removeRegion(int id) { 
		if (regions.isEmpty()) return false;
		
		boolean notDone = true;
		int i = 0;
		
		while(i < regions.size() && notDone) {
			if(regions.get(i).id == id) {
				notDone = false;
			} else {
				i++;
			}
		}
		
		if(! notDone) {
			regions.remove(i);
			return true;
		} else {
			return false;
		}
	}
	
	public void render(Graphics g) {
		for(Region r : regions) {
			r.render(g);
		}
		
	}
	
}
