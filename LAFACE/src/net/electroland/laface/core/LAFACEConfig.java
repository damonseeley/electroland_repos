package net.electroland.laface.core;

import java.util.Vector;

import net.electroland.laface.util.SensorPair;

public class LAFACEConfig {
	public static boolean TESTING = false;
	
	// for car sprites
	final public static int NORTH = 0;
	final public static int SOUTH = 1;
	
	// the upper limit is the longest total amount of time that an object can take
	// to cross the screen.  this is AFTER the time multipler is taking into affect.
	// if an object is calculated to take any longer than this, it is instead assigned
	// this value as it's crossing time.
	final public static long UPPER_TIME_LIMIT = 9000;
	// similar to upper, but defines the lowest limit.
	final public static long LOWER_TIME_LIMIT = 4000;
	
	final public static String CAMERA_ELV_FNAME = "./noho_north_vs01.elv";
	
	public Vector<SensorPair> NORTH_SENSOR_PAIRS = null;
	public Vector<SensorPair> SOUTH_SENSOR_PAIRS = null;
	final public static int CARSTRIGGERTIMEOUT = 4 * 1000; // the amount of time between each triggering car before the counter resets
		
	public LAFACEConfig() {
		// each Sensor pair has 5 arguments.
		// SensorPair(id1, id2, threshold, TYPE, laneId, tmultiplier)
		// id1 = the Elvis ID of the 'start' sensor.
		// id2 = the Elvis ID of the 'finish' sensor.  this is the sensor that starts the sprite
		// threshold is the time threshold over which the sensor will throw out the detected object (milliseconds between sensors)
		// TYPE determines what kind of sprite is started. right now there are only NORTH and SOUTH.
		// laneId is just the ID output used for System.out.
		// tmultiplier- multiply the time by this to specify how many seconds it takes the object to travel the width of the screen.

		NORTH_SENSOR_PAIRS = new Vector<SensorPair>();
		NORTH_SENSOR_PAIRS.add(new SensorPair(8, 10, 1500, NORTH, 0, 28.0));
		NORTH_SENSOR_PAIRS.add(new SensorPair(11, 14, 1500, NORTH, 1, 21.0));

		SOUTH_SENSOR_PAIRS = new Vector<SensorPair>();
		SOUTH_SENSOR_PAIRS.add(new SensorPair(0, 1, 5000, SOUTH, 0, 12.0));
		SOUTH_SENSOR_PAIRS.add(new SensorPair(10, 14, 5000, SOUTH, 1, 10.0));
	}
}
