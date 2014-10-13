package net.electroland.enteractive.core;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.udpUtils.UDPParser;
import net.electroland.util.NoDataException;
import net.electroland.util.RunningAverage;

import org.apache.log4j.Logger;

public class Model {
	
	static Logger logger = Logger.getLogger(Model.class);
	private int personIndex = 0;
	private ConcurrentHashMap<Integer,Person> people;	// all active people
	private List <ModelListener> listeners;
	private boolean[] sensors, pastSensors;
	private int gridWidth;
	private boolean fourCornersOn = false;
	private boolean oppositeCornersOn = false;
	private boolean oppositeCorners2On = false;
	private boolean empty = false;
	private RunningAverage average;
	private int numSamples, sampleRate;
	
	public Model(int gridWidth, int gridHeight){
		this.gridWidth = gridWidth;
		people = new ConcurrentHashMap<Integer,Person>();
		listeners = new ArrayList<ModelListener>();
		sensors = new boolean[gridWidth*gridHeight];
		pastSensors = new boolean[sensors.length];
		for(int i=0; i<sensors.length; i++){
			sensors[i] = false;
		}

		// This will store and calculate running averages.
		numSamples = 10;
		sampleRate = 33;	// millis
		average = new RunningAverage(numSamples);
	}
	
	public void updateSensors(int offset, boolean[] data, Map<Integer, Tile>stuckTiles){
		boolean[] olddata = new boolean[data.length];
		try{
			//System.out.println(offset+" "+sensors.length+" "+olddata.length);
			System.arraycopy(sensors, offset, olddata, 0, olddata.length);	// past states of currently reporting sensors
			compareSensorStates(offset, data, olddata);						// look for any on/off event activity
			
			// 2014 addition to prevent stuck tiles from generating people
			removePeopleFromStuckSensors(stuckTiles);
			
			System.arraycopy(sensors, 0, pastSensors, 0, sensors.length);	// copy all past sensor states
			System.arraycopy(data, 0, sensors, offset, data.length);		// paste in new sensor states
			checkForEvents();
		} catch(ArrayIndexOutOfBoundsException e){
			logger.error("error updating sensor states. offset: "+offset);
			logger.error(e);
		}
	}
	
	private void compareSensorStates(int offset, boolean[] newstates, boolean[] oldstates){
		for(int i=0; i<newstates.length; i++){							// for each sensor...
			if(!oldstates[i] && newstates[i]){							// if sensor just turned on...
				int x = (i % gridWidth) + (offset % gridWidth);
				int y = offset / gridWidth;
				Person newperson = new Person(personIndex, offset+i, x+1, y);	// new person at grid x/y
				synchronized(people){
					people.put(personIndex, newperson);						// add to master people map
				}
				//System.out.println(personIndex +" added");
				personIndex++;
			} else if(oldstates[i] && !newstates[i]){
				synchronized(people){
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
						person.die();
						people.remove(person.getID());						// remove person from active people
						//if(!people.containsKey(person.getID())){
							//System.out.println(person.getID() +" removed");
							//System.out.println("people size: "+people.size());
						//}
					}
				}
			}
		}
	}
	
	
	/*
	 * 2014 addition
	 * immediately after people are processed, removes people from tiles that are "stuck"
	 * such that the do not affect shows or the runnign average for screensaver output
	 */
	private void removePeopleFromStuckSensors(Map<Integer, Tile>stuckTiles){
		synchronized(people){
			Iterator<Person> iter = people.values().iterator();
			Person person = null;
			while(iter.hasNext()){	// for each active person...
				person = iter.next();
				if(isStuck(person.getLinearLoc() + 1, stuckTiles)){	// if person is on a stuck tile
					logger.info("KILLING PERSON " + person.getLinearLoc() + " DUE TO STUCK TILE ");
					person.die();	// kill
					people.remove(person.getID()); // and remove person from active people
				}
			}
		}
	}

	
	private static boolean isStuck(int i, Map<Integer, Tile> stuckTiles){
		//System.out.println("testing for stuck on " + i + stuckTiles);
		return stuckTiles.get(i)!=null;
	}
	
	
	private void checkForEvents(){
		if(sensors[0] && sensors[15] && sensors[160] && sensors[175]){				// if all 4 corners are active...
			if(!fourCornersOn){
				ModelEvent event = new ModelEvent(ModelConstants.FOUR_CORNERS);
				fourCornersOn = true;
				notifyListeners(event);
			}
		} else{
			fourCornersOn = false;
			if(sensors[0] && sensors[175]){										// opposite corner
				if(!oppositeCornersOn){
					ModelEvent event = new ModelEvent(ModelConstants.OPPOSITE_CORNERS);
					oppositeCornersOn = true;
					notifyListeners(event);
				}
			} else {
				oppositeCornersOn = false;
			}
			if(sensors[15] && sensors[160]){									// opposite corners
				if(!oppositeCorners2On){
					ModelEvent event = new ModelEvent(ModelConstants.OPPOSITE_CORNERS2);
					oppositeCorners2On = true;
					notifyListeners(event);
				}
			} else {
				oppositeCorners2On = false;
			}
		}		
	}
	
	public void addListener(ModelListener listener){
		listeners.add(listener);
	}
	
	public void removeListener(ModelListener listener){
		listeners.add(listener);
	}
	
	public void notifyListeners(ModelEvent event){
		Iterator<ModelListener> i = listeners.iterator();
		while (i.hasNext()){
			i.next().modelEvent(event);
		}
	}
	
	public boolean[] getSensors(){
		return sensors;
	}
	
	public ConcurrentHashMap<Integer,Person> getPeople(){
		return people;
	}

	public void updateAverage(double averageSample){
		average.addValue(averageSample, sampleRate);
		
		try {
			if(average.getAvg() == 0.0 && !empty){
				ModelEvent event = new ModelEvent(ModelConstants.EMPTY);
				empty = true;
				notifyListeners(event);
			} else if(average.getAvg() > 0.0 && empty){
				ModelEvent event = new ModelEvent(ModelConstants.NOT_EMPTY);
				empty = false;
				notifyListeners(event);
			}
		} catch (NoDataException e) {
			//e.printStackTrace();
		}
	}
	
	public double getAverage() throws NoDataException{
		return average.getAvg();
	}
	
	
	
	
	
	
	public class ModelConstants{
		static public final int FOUR_CORNERS = 1;
		static public final int OPPOSITE_CORNERS = 2;
		static public final int OPPOSITE_CORNERS2 = 3;
		static public final int EMPTY = 10;		// sent when average drops to 0
		static public final int NOT_EMPTY = 11;	// sent when average goes from 0 to anything
	}
	

}
