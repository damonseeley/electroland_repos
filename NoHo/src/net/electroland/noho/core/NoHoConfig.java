package net.electroland.noho.core;

import java.util.Vector;

import net.electroland.noho.core.fontMatrix.FontMatrix;
import net.electroland.noho.util.SensorPair;

public class NoHoConfig {
	
	//testing
	public static boolean TESTING = false;
	
	public static String QUOTEFILE = "./afi100_photoshoot_quotes.txt";
	//public static String QUOTEFILE = "./afi100_quotes_list_modified_with_breaks.txt";
	
	public static String FONT_DIRECTORY ="5x7_STD_GIF/";
	final public static int FRAMERATE = 50;
	final public static int CHARWIDTH = 10;
	final public static int CHARHEIGHT = 14;
	final public static int DISPLAYWIDTH = 390;
	final public static int DISPLAYHEIGHT = 14;
	final public static int TOTALCHARS = DISPLAYWIDTH / CHARWIDTH;
	
	//timing
	final public static int PHRASETIMING = 8 * 1000;
	final public static int TEXTTIMEOUT = PHRASETIMING * 4 - 1000; //4 = 4 LINES, REGARDLESS OF PHRASES // not used in new system
	// TRY 75s for SPRITES
	final public static int SPRITESMAXTIME = 90 * 1000; // max time that sprite mode can be active
	final public static int SPRITESMINTIME = 15 * 1000;  // min time that sprites can be active before timing out to text
	final public static int SPRITESTIMEOUT = 12 * 1000; // period during which if no cars appear the app will switch to text mode
	
	final public static int CARSTRIGGERTHRESH = 3; // the number of total cars that will cause a switch from text to sprite mode
	final public static int CARSTRIGGERTIMEOUT = 4 * 1000; // the amount of time between each triggering car before the counter resets
	//final public static int CARSTRIGGERTIMEOUT = 1000 * 1000; // the amount of time between each triggering car before the counter resets

	
	
	//environmental
	final public static int MAXTEMP = 100;
	final public static int LOWCONDITION = 29;
	final public static double LOWVISIBILITY = 8.0;

	// for car sprites
	final public static int NORTH = 0;
	final public static int SOUTH = 1;
//	final public static long NORTH_SENSOR_THRESHOLD = 1500;
//	final public static long SOUTH_SENSOR_THRESHOLD = 5000; // 
	final public static double NORTH_INIT_X_OFFSET = 320.0;
	final public static double SOUTH_INIT_X_OFFSET = 20.0; // in pixels, from the left or right respectively
	final public static String NORTH_CAMERA_ELV_FNAME = "./noho_north_vs01.elv";
	final public static String SOUTH_CAMERA_ELV_FNAME = "./noho_south_vs01.elv";

	// explosion sprites
	final public static int BLAST_WIDTH = 30;
	final public static int BLAST_HEIGHT = 14;
	final public static int NUM_CONFETTI = 50;
	
	// the upper limit is the longest total amount of time that an object can take
	// to cross the screen.  this is AFTER the time multipler is taking into affect.
	// if an object is calculated to take any longer than this, it is instead assigned
	// this value as it's crossing time.
	final public static long UPPER_TIME_LIMIT = 9000;
	// similar to upper, but defines the lowest limit.
	final public static long LOWER_TIME_LIMIT = 4000;
	
	
	public Vector<SensorPair> NORTH_SENSOR_PAIRS = null;
	public Vector<SensorPair> SOUTH_SENSOR_PAIRS = null;
	
	final public static FontMatrix fontStandard = new FontMatrix(NoHoConfig.FONT_DIRECTORY);
	
	//TESTING VARS
	public static String[] TEMPMESSAGES;
	
	public NoHoConfig() {
		
	//	each Sensor pair has 5 arguments.
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
