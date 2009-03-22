package net.electroland.enteractive.core;

import java.util.HashMap;
import java.util.Iterator;

public class Model {
	
	private int personIndex = 0;
	private HashMap<Integer,Person> people;	// all active people
	//private HashMap<Integer,Person> enters;	// brand new people
	//private HashMap<Integer,Person> exits;		// people who just left
	private boolean[] sensors, pastSensors;
	private int gridWidth;
	
	public Model(int gridWidth, int gridHeight){
		this.gridWidth = gridWidth;
		people = new HashMap<Integer,Person>();
		//enters = new HashMap<Integer,Person>();
		//exits = new HashMap<Integer,Person>();
		sensors = new boolean[gridWidth*gridHeight];
		pastSensors = new boolean[sensors.length];
		for(int i=0; i<sensors.length; i++){
			sensors[i] = false;
		}
	}
	
	public void updateSensors(int offset, boolean[] data){
		boolean[] olddata = new boolean[data.length];
		System.arraycopy(sensors, offset, olddata, 0, olddata.length);	// past states of currently reporting sensors
		compareSensorStates(offset, data, olddata);						// look for any on/off event activity
		System.arraycopy(sensors, 0, pastSensors, 0, sensors.length);	// copy all past sensor states
		System.arraycopy(data, 0, sensors, offset, data.length);		// paste in new sensor states
	}
	
	public void compareSensorStates(int offset, boolean[] newstates, boolean[] oldstates){
		for(int i=0; i<newstates.length; i++){							// for each sensor...
			if(!oldstates[i] && newstates[i]){							// if sensor just turned on...
				int x = (i % gridWidth) + (offset % gridWidth);
				int y = offset / gridWidth;
				Person newperson = new Person(personIndex, offset+i, x+1, y);	// new person at grid x/y
				people.put(personIndex, newperson);						// add to master people map
				//enters.put(personIndex, newperson);						// add to new enters map
				//System.out.println(personIndex +" added");
				personIndex++;
			} else if(oldstates[i] && !newstates[i]){
				Iterator<Person> iter = people.values().iterator();
				int id = -1;
				Person person = null;
				// TODO this seems fairly kludge... any better ideas?
				while(iter.hasNext()){									// for each active person...
					person = iter.next();
					if(person.getLinearLoc() == offset+i){				// if person is on the tile that just turned off...
						id = person.getID();							// get their id
						break;											// exit the loop
					}
				}
				if(id >= 0){
					//exits.put(person.getID(), person);					// add person to the exits map
					person.die();
					people.remove(person.getID());						// remove person from active people
					if(!people.containsKey(person.getID())){
						//System.out.println(person.getID() +" removed");
						//System.out.println("people size: "+people.size());
					}
				}
			}
		}
	}
	
	public boolean[] getSensors(){
		return sensors;
	}
	
	public HashMap<Integer,Person> getPeople(){
		return people;
	}
	
	/*
	public HashMap<Integer,Person> getEnters(){
		// this is for shows to know when to instantiate a sprite
		return enters;	// should be cleared by the show that calls it
	}
	
	public HashMap<Integer,Person> getExits(){
		// this is for shows to know when to destroy a sprite
		return exits;	// should be cleared by the show that calls it
	}
	
	public void clearEnters(){
		enters.clear();
	}
	
	public void clearExits(){
		exits.clear();
	}
	*/

}
