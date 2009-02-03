package net.electroland.lafm.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import org.apache.log4j.Logger;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;
import net.electroland.lafm.gui.GUI;
import net.electroland.lafm.gui.GUIWindow;
import net.electroland.lafm.scheduler.TimedEvent;
import net.electroland.lafm.scheduler.TimedEventListener;
import net.electroland.lafm.shows.AdditivePropellerThread;
import net.electroland.lafm.shows.ChimesThread;
import net.electroland.lafm.shows.DartBoardThread;
import net.electroland.lafm.shows.FireworksThread;
import net.electroland.lafm.shows.GlobalColorShift;
import net.electroland.lafm.shows.Glockenspiel;
import net.electroland.lafm.shows.ImageSequenceThread;
import net.electroland.lafm.shows.KnockoutThread;
import net.electroland.lafm.shows.LightGroupTestThread;
import net.electroland.lafm.shows.PieThread;
import net.electroland.lafm.shows.PropellerThread;
import net.electroland.lafm.shows.RadialWipeThread;
import net.electroland.lafm.shows.RandomPropellerThread;	// using this for now
import net.electroland.lafm.shows.ShutdownThread;
import net.electroland.lafm.shows.SparkleSpiralThread;
import net.electroland.lafm.shows.SpinningRingThread;
import net.electroland.lafm.shows.SpinningThread;
import net.electroland.lafm.shows.SpiralThread;
import net.electroland.lafm.shows.ThrobbingThread;
import net.electroland.lafm.shows.VegasThread;
import net.electroland.lafm.shows.WipeThread;
import net.electroland.lafm.weather.WeatherChangeListener;
import net.electroland.lafm.weather.WeatherChangedEvent;
import net.electroland.lafm.weather.WeatherChecker;
import net.electroland.lafm.util.ColorScheme;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import promidi.MidiIO;
import promidi.Note;

public class Conductor extends Thread implements ShowThreadListener, WeatherChangeListener, TimedEventListener, ShowCollectionListener{

	static Logger logger = Logger.getLogger(Conductor.class);
	
	public GUIWindow guiWindow;				// frame for GUI
	//static public DMXLightingFixture[] flowers;		// all flower fixtures
	public MidiIO midiIO;						// sensor data IO
	public DetectorManager detectorMngr;
	public SoundManager soundManager;
	public WeatherChecker weatherChecker;
	public Properties sensors;					// pitch to fixture mappings
	public Properties systemProps;
	public Properties physicalProps;
	public TimedEvent[] clockEvents;
	private ImageSequenceCache imageCache; 	// for ImageSequenceThreads
	public String[] sensorShows, timedShows;	// list of names of sensor-triggered shows
	public String[] fixtureActivity;			// 22 fixtures, null if empty; show name if in use
	public String[] physicalColors;
	public int currentSensorShow;				// number of show to display when sensor is triggered
	public boolean forceSensorShow = false;
	public boolean headless;
	public String[] floors = new String[3];
	public int[] sectors = new int[9];
	public int width, height;

	// sample timed events, but I assume the building will be closed for some time at night
	//TimedEvent sunriseOn = new TimedEvent(6,00,00, this); // on at sunrise-1 based on weather
	//TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather

	private List <ShowThread>liveShows;
	private List <DMXLightingFixture> availableFixtures;
	private List <DMXLightingFixture> fixtures;
	
	// Images used in processing based procedural shows should be loaded BEFORE the show is instantiated,
	// otherwise multiple shows going off at once will have delays in between play back.
	private PImage innerRing, outerRing;
	private PImage innerRingRed, outerRingRed;
	private PImage innerRingOrange, outerRingOrange;
	private PImage innerRingYellow, outerRingYellow;
	private PImage innerRingPink, outerRingPink;
	private PImage innerRingPurple, outerRingPurple;
	private PImage sweepSprite;
	private PImage sweepRed, sweepOrange, sweepYellow, sweepPink, sweepPurple;
	private PImage[] sweepSpriteList;

	public Conductor(String args[]){
		
		try{
			systemProps = new Properties();
			systemProps.load(new FileInputStream(new File("depends//system.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try{
			physicalProps = new Properties();
			physicalProps.load(new FileInputStream(new File("depends//physical.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
			
		Properties lightProps = new Properties();
		try {

			// load props
			lightProps.load(new FileInputStream(new File("depends//lights.properties")));

			// set up fixtures
			detectorMngr = new DetectorManager(lightProps);

			// get fixtures
			fixtures = Collections.synchronizedList(new ArrayList <DMXLightingFixture>(detectorMngr.getFixtures()));
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		headless = Boolean.parseBoolean(systemProps.getProperty("headless"));

		// to track which fixtures are used, and what shows are currently running.
		liveShows = Collections.synchronizedList(new ArrayList<ShowThread>());
		availableFixtures = Collections.synchronizedList(new ArrayList<DMXLightingFixture>(detectorMngr.getFixtures()));
		
		width = availableFixtures.get(0).getWidth();
		height = availableFixtures.get(0).getWidth();
		
		// define vertical positions of fixtures
		floors[0] = "lower";
		floors[1] = "middle";
		floors[2] = "upper";
		
		// define radially oriented positions of fixtures
		for(int i=0; i<sectors.length; i++){
			sectors[i] = i;
		}
		
		currentSensorShow = 0;
		sensorShows = new String[15];		// size dependent on number of sensor-triggered shows
		sensorShows[0] = "Throb";
		sensorShows[1] = "Propeller";
		sensorShows[2] = "Spiral";
		sensorShows[3] = "Dart Board";
		sensorShows[4] = "Pie";
		sensorShows[5] = "Flashing Pie";
		sensorShows[6] = "Vegas";
		sensorShows[7] = "Fireworks";
		sensorShows[8] = "Additive Propeller";
		sensorShows[9] = "Light Group Test";
		sensorShows[10] = "Gradient Rings";
		sensorShows[11] = "Wipe";
		sensorShows[12] = "Knockout";
		sensorShows[13] = "Sweep";
		sensorShows[14] = "Radial Wipe";
		
		timedShows = new String[8];
		timedShows[0] = "Solid Color";
		timedShows[1] = "Dart Boards";
		timedShows[2] = "Sparkle Spiral";
		timedShows[3] = "Gradient Rings";
		timedShows[4] = "Vertical Color Shift";
		timedShows[5] = "Radial Color Shift";
		timedShows[6] = "Fireworks";
		timedShows[7] = "Sweeps";
		
		physicalColors = new String[5];
		physicalColors[0] = "red";
		physicalColors[1] = "orange";
		physicalColors[2] = "yellow";
		physicalColors[3] = "purple";
		physicalColors[4] = "pink";
		
		sensors = new Properties();
		try{

			// load sensor info
			sensors.load(new FileInputStream(new File("depends//sensors.properties")));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		clockEvents = new TimedEvent[24*12];		// event every 5 minutes
		for(int h=0; h<24; h++){
			for(int m=0; m<12; m++){
				clockEvents[(h+1)*m] = new TimedEvent(h,m*5,0,this);
			}
		}
		
		/*
		clockEvents = new TimedEvent[24*60];	// event every minute just for testing
		for(int h=0; h<24; h++){
			for(int m=0; m<60; m++){
				clockEvents[(h+1)*m] = new TimedEvent(h,m,0,this);
			}
		}
		*/

		midiIO = MidiIO.getInstance();
		midiIO.printDevices();
		try{
			midiIO.plug(this, "midiEvent", Integer.parseInt(systemProps.getProperty("midiDeviceNumber")), 0);	// device # and midi channel
		} catch(Exception e){
			e.printStackTrace();
		}
		soundManager = new SoundManager(systemProps.getProperty("soundAddress"), Integer.parseInt(systemProps.getProperty("soundPort")));
		
		// GUI must be instantiated for use by ImageSequenceCache
		guiWindow = new GUIWindow(this, detectorMngr.getDetectors());
		if(!headless){
			guiWindow.setVisible(true);
		}
		
		try {
			Properties imageProps = new Properties();
			imageProps.load(new FileInputStream(new File("depends//images.properties")));
			imageCache = new ImageSequenceCache(imageProps, guiWindow.gui);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		innerRingRed = guiWindow.gui.loadImage("depends//images//sprites//innerRing_white_red_yellow.png");
		outerRingRed = guiWindow.gui.loadImage("depends//images//sprites//outerRing_white_red_yellow.png");
		innerRingOrange = guiWindow.gui.loadImage("depends//images//sprites//innerRing_red_yellow_black.png");
		outerRingOrange = guiWindow.gui.loadImage("depends//images//sprites//outerRing_red_yellow_black.png");
		innerRingYellow = guiWindow.gui.loadImage("depends//images//sprites//innerRing_blue_green_black.png");
		outerRingYellow = guiWindow.gui.loadImage("depends//images//sprites//outerRing_blue_green_black.png");
		innerRingPink = guiWindow.gui.loadImage("depends//images//sprites//innerRing_red_purple_cyan.png");
		outerRingPink = guiWindow.gui.loadImage("depends//images//sprites//outerRing_red_purple_cyan.png");
		innerRingPurple = guiWindow.gui.loadImage("depends//images//sprites//innerRing_blue_purple_black.png");
		outerRingPurple = guiWindow.gui.loadImage("depends//images//sprites//outerRing_blue_purple_black.png");
		
		sweepRed = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_red_yellow.png");
		sweepOrange = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_quads_rgb_light.png");
		sweepYellow = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_watermelon.png");
		sweepPink = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_cyan_pink.png");
		sweepPurple = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_lizard.png");
		
		sweepSpriteList = new PImage[13];
		sweepSpriteList[0] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_red_yellow.png");
		sweepSpriteList[1] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_blue_orange.png");
		sweepSpriteList[2] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_cyan_pink.png");
		sweepSpriteList[3] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_lizard.png");
		sweepSpriteList[4] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_pink.png");
		sweepSpriteList[5] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_purple_green.png");
		sweepSpriteList[6] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_quads_rgb_light.png");
		sweepSpriteList[7] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_quads_rgb.png");
		sweepSpriteList[8] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_red_yellow.png");
		sweepSpriteList[9] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_watermelon.png");
		sweepSpriteList[10] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_red_aqua.png");
		sweepSpriteList[11] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_strobe.png");
		sweepSpriteList[12] = guiWindow.gui.loadImage("depends//images//sprites//sweeps//sweep_white.png");
		
		
		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
		//weatherChecker.start();

		Runtime.getRuntime().addShutdownHook(new ShutdownThread(fixtures, guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P2D), "ShutdownShow"));
	
	}
	
	public void midiEvent(Note note){

		// is it an on or off event?
		boolean on = note.getVelocity() == 0 ? false : true;
		
		// get the name of the fixture tied to that note value
		String fixtureId = sensors.getProperty(String.valueOf(note.getPitch()));

		// find the actual fixture
		DMXLightingFixture fixture = detectorMngr.getFixture(fixtureId);
				
		// did we get a fixture?
		if (fixture != null){
				
			// tell any show thread that is a midi listener that an event occured.
			Iterator<ShowThread> i = liveShows.iterator();
				while (i.hasNext()){
				ShowThread s = i.next();
				if (s instanceof SensorListener){
					((SensorListener)s).sensorEvent(fixture, on);
				}
			}
			
			if (on){
				// on events
				PGraphics raster = guiWindow.gui.createGraphics(fixture.getWidth(), fixture.getHeight(), PConstants.P3D);
				ShowThread newShow = new ThrobbingThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 500, 500, 0, 0, 0, 0, "none", physicalProps, 1);	// default
				
				if(forceSensorShow){
					
					// THIS IS ONLY RUN WHEN DEFAULT SENSOR SHOWS ARE OVERRIDDEN BY GUI SELECTION
					
					float[][] colorlist;
					float[] pointlist;
					ColorScheme spectrum;
					
					switch (currentSensorShow) {
			            case 0:
			            	newShow = new ThrobbingThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 250, 250, 0, 0, 0, 0, "none", physicalProps, 1);
			            	break;
			            case 1:
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
			            	newShow = new PropellerThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, spectrum, 20, 5, 0.1f, 0.1f, "none", physicalProps, 1);
			            	break;
			            case 2:
			            	newShow = new SpiralThread(fixture, soundManager, 10, detectorMngr.getFps(), raster, "SpiralThread", ShowThread.LOW, 0, 255, 255, 30, 2, 3, 100, guiWindow.gui.loadImage("depends//images//sprites//thicksphere50alpha.png"), "none", physicalProps, 1);
			            	break;
			            case 3: 
							colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
							newShow = new DartBoardThread(fixture, soundManager, 7, detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.LOW, spectrum, 0.01f, 0.1f, 0.001f, 0.002f, 0.15f, "none", physicalProps, true, guiWindow.gui.loadImage("depends//images//sprites//randomwhitespots.png"), 1);
							break;
			            case 4:
			            	newShow = new PieThread(fixture, soundManager, 3, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255, 255, 0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"), "none", physicalProps, 1);
			            	break;
			            case 5:
			            	newShow = new PieThread(fixture, soundManager, 2, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255,255,0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"), "none", physicalProps, 1);
							newShow.chain(new ThrobbingThread(fixture, soundManager, 1, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255,255,0, 100, 100, 0, 0, 0, 0, "none", physicalProps, 1));									
							break;
			            case 6:
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
							newShow = new VegasThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "VegasThread", ShowThread.LOW, spectrum, 0, 0 , 0.5f, 0.001f, "none", physicalProps, 1);
							//newShow = new ExpandingThread(fixture, null,2, detectorMngr.getFps(), raster, "ExpandingThread", ShowThread.LOW, guiWindow.gui.loadImage("depends//images//sprites//sphere50alpha.png"));
							//newShow.chain(new PieThread(fixture, null, 2, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255, 255, 0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png")));
							//newShow.chain(new ImageSequenceThread(fixture, null, 2, detectorMngr.getFps(), raster, "BubblesThread", ShowThread.LOW, imageCache.getSequence("redThrob"), false));		
							break;
			            case 7:
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
							newShow = new FireworksThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "FireworksThread", ShowThread.LOW, spectrum, 8, 0.8f, guiWindow.gui.loadImage("depends//images//sprites//thickring.png"), "none", physicalProps, 0, true, 1);
							break;
			            case 8: 
			            	newShow = new AdditivePropellerThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "AdditivePropellerThread", ShowThread.LOW, "255:0:0", "0:255:0", "0:0:255", 3, 5, 0.2f, 0.5f, "none", physicalProps, 1);
			            	break;
			            case 9:
			            	// light group test
			            	newShow = new LightGroupTestThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "LightGroupTest", ShowThread.LOW, guiWindow.gui.loadImage("depends//images//lightgrouptest2.png"));
			            	break;
			            case 10:
			            	// gradient rings
			            	if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
								innerRing = innerRingRed;
								outerRing = outerRingRed;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
								innerRing = innerRingOrange;
								outerRing = outerRingOrange;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
								innerRing = innerRingYellow;
								outerRing = outerRingYellow;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
								innerRing = innerRingPink;
								outerRing = outerRingPink;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
								innerRing = innerRingPurple;
								outerRing = outerRingPurple;
							}
			            	newShow = new SpinningRingThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "GradientRings", ShowThread.LOW, 255, 255, 255, 2, 7, 20, 5, 0.05f, 0.05f, 0.05f, 0.05f, outerRing, innerRing, false, "none", physicalProps, 0, true, 1);
			            	break;
			            case 11:
			            	// wipe
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
			            	newShow = new WipeThread(fixture, soundManager, 10, detectorMngr.getFps(), raster, "Wipe", ShowThread.LOW, spectrum, 5, 5, 2, "none", physicalProps, 1);
			            	break;
			            case 12:
			            	// knockout
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
			            	newShow = new KnockoutThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "Knockout", ShowThread.LOW, spectrum, 10, 5, 0.01f, 0, 0.5f, 0.001f, "none", physicalProps, 1);
			            	//newShow = new ImageSequenceThread(fixture, soundManager, 5, detectorMngr.getFps(), raster, "Knockout", ShowThread.LOW, imageCache.getSequence("knockout"), false, "none", physicalProps);					
			            	break;
			            case 13:
			            	// sweeps
			            	if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
								sweepSprite = sweepRed;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
								sweepSprite = sweepOrange;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
								sweepSprite = sweepYellow;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
								sweepSprite = sweepPink;
							} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
								sweepSprite = sweepPurple;
							}
			            	newShow = new SpinningThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "Sweep", ShowThread.LOW, sweepSprite, 2, 0.1f, 0.2f, 5, "none", physicalProps, 0, true, 1);
			            	//newShow = new SpinningRingThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "GradientRings", ShowThread.LOW, 255, 255, 255, 2, 7, 20, 5, 0.05f, 0.05f, 0.05f, 0.05f, outerRing, innerRing, false, "none", physicalProps);
			            	break;
			            case 14:
			            	// radial wipe
			            	colorlist = new float[3][3];
							colorlist[0][0] = 255;
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
							colorlist[2][0] = 255;
							colorlist[2][1] = 0;
							colorlist[2][2] = 0;
							pointlist = new float[3];
							pointlist[0] = 0;
							pointlist[1] = 0.5f;
							pointlist[2] = 1;
							spectrum = new ColorScheme(colorlist, pointlist);
			            	newShow = new RadialWipeThread(fixture, soundManager, 10, detectorMngr.getFps(), raster, "Wipe", ShowThread.LOW, spectrum, 5, 5, 10, "none", physicalProps, 1);
			            	break;
			            	
					}
					
					
				} else {
					String[] showProps = systemProps.getProperty(fixtureId).split(",");
					
					if(showProps[0].equals("propeller")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						//newShow = new PropellerThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, spectrum, Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), showProps[8], physicalProps);
						newShow = new RandomPropellerThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, spectrum, Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), showProps[8], physicalProps, Float.parseFloat(showProps[9]));
					} else if(showProps[0].equals("throb")){
						newShow = new ThrobbingThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), Integer.parseInt(showProps[7]), Integer.parseInt(showProps[8]), Integer.parseInt(showProps[9]), Integer.parseInt(showProps[10]), showProps[11], physicalProps, Float.parseFloat(showProps[12]));
					} else if(showProps[0].equals("spiral")){
						PImage sprite = null;
		            	if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							sprite = guiWindow.gui.loadImage("depends//images//sprites//blueRedSphere.png");
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							sprite = guiWindow.gui.loadImage("depends//images//sprites//yellowRedSphere.png");
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							sprite = guiWindow.gui.loadImage("depends//images//sprites//yellowCyanSphere.png");
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							sprite = guiWindow.gui.loadImage("depends//images//sprites//blueRedSphere.png");
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							sprite = guiWindow.gui.loadImage("depends//images//sprites//blueRedSphere.png");
						}
						newShow = new SpiralThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "SpiralThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), Integer.parseInt(showProps[7]), Integer.parseInt(showProps[8]), sprite, showProps[9], physicalProps, Float.parseFloat(showProps[10]));
					} else if(showProps[0].equals("dartboard")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new DartBoardThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), Float.parseFloat(showProps[8]), showProps[9], physicalProps, true, guiWindow.gui.loadImage("depends//images//sprites//randomwhitespots.png"), Float.parseFloat(showProps[10]));
					} else if(showProps[0].equals("images")){
						newShow = new ImageSequenceThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, showProps[2], ShowThread.LOW, imageCache.getSequence(showProps[2]), false, showProps[3], physicalProps, Float.parseFloat(showProps[5]));					
						((ImageSequenceThread)newShow).enableTint(Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]));
					} else if(showProps[0].equals("vegas")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new VegasThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "VegasThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), showProps[8], physicalProps, Float.parseFloat(showProps[9]));
					} else if(showProps[0].equals("fireworks")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new FireworksThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "FireworksThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), guiWindow.gui.loadImage("depends//images//sprites//thickring.png"), showProps[6], physicalProps, 0, true, Float.parseFloat(showProps[7]));
					} else if(showProps[0].equals("additivepropeller")){
						newShow = new AdditivePropellerThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "AdditivePropellerThread", ShowThread.LOW, showProps[2], showProps[3], showProps[4], Float.parseFloat(showProps[5]), Integer.parseInt(showProps[6]), Float.parseFloat(showProps[7]), Float.parseFloat(showProps[8]), showProps[9], physicalProps, Float.parseFloat(showProps[10]));
					} else if(showProps[0].equals("flashingpie")){
						newShow = new PieThread(fixture, soundManager, Integer.parseInt(showProps[5]), detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"), showProps[5], physicalProps, Float.parseFloat(showProps[6]));
						newShow.chain(new ThrobbingThread(fixture, soundManager, Integer.parseInt(showProps[6]), detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), 300, 300, 0, 0, 0, 0, showProps[5], physicalProps, Float.parseFloat(showProps[6])));									
					} else if(showProps[0].equals("gradientrings")){
						if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							innerRing = innerRingRed;
							outerRing = outerRingRed;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							innerRing = innerRingOrange;
							outerRing = outerRingOrange;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							innerRing = innerRingYellow;
							outerRing = outerRingYellow;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							innerRing = innerRingPink;
							outerRing = outerRingPink;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							innerRing = innerRingPurple;
							outerRing = outerRingPurple;
						}
		            	newShow = new SpinningRingThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "GradientRings", ShowThread.LOW, 255, 255, 255, Float.parseFloat(showProps[2]), Float.parseFloat(showProps[3]), Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), Float.parseFloat(showProps[8]), Float.parseFloat(showProps[9]), outerRing, innerRing, Boolean.parseBoolean(showProps[10]), showProps[11], physicalProps, 0, true, Float.parseFloat(showProps[12]));
		            } else if(showProps[0].equals("wipe")){
		            	ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new WipeThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "Wipe", ShowThread.LOW, spectrum, Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]),showProps[7], physicalProps, Float.parseFloat(showProps[8]));
					} else if(showProps[0].equals("knockout")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new KnockoutThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "Knockout", ShowThread.LOW, spectrum, Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]), Float.parseFloat(showProps[8]), Float.parseFloat(showProps[9]), showProps[10], physicalProps, Float.parseFloat(showProps[11]));
					} else if(showProps[0].equals("radialwipe")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
		            	newShow = new RadialWipeThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "RadialWipe", ShowThread.LOW, spectrum, Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), showProps[7], physicalProps, Float.parseFloat(showProps[8]));
					} else if(showProps[0].equals("sweep")){
						// sweeps
						int luckynumber = (int)(Math.random()*(sweepSpriteList.length-1));
						sweepSprite = sweepSpriteList[luckynumber];
		            	newShow = new SpinningThread(fixture, soundManager, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "Sweep", ShowThread.LOW, sweepSprite, Float.parseFloat(showProps[2]), Float.parseFloat(showProps[3]), Float.parseFloat(showProps[4]), Integer.parseInt(showProps[5]), showProps[6], physicalProps, 0, true, Float.parseFloat(showProps[7]));
					}
				}
				
				// everything happens in here now.
				startShow(newShow);				
			}				
		}
	}
	
	public ColorScheme processColorScheme(String colordata, String pointdata){
		// red:green:blue-red:green:blue, point-point-point
		String[] colors = colordata.split("-");
		String[] points = pointdata.split("-");
		float[][] colorlist = new float[colors.length][3];
		float[] pointlist = new float[points.length];
		for(int n=0; n<points.length; n++){
			pointlist[n] = Float.parseFloat(points[n]);
			String[] tempcolor = colors[n].split(":");
			colorlist[n][0] = Float.parseFloat(tempcolor[0]);
			colorlist[n][1] = Float.parseFloat(tempcolor[1]);
			colorlist[n][2] = Float.parseFloat(tempcolor[2]);
		}
		return new ColorScheme(colorlist, pointlist);
	}
	
	public List <ShowThread> getLiveShows(){
		return liveShows;
	}
	
	//******************** THREAD MANAGEMENT ***********************************
	// Any time a show is done, this call back will be called, so that
	// conductor knows the show is over, and that the flowers are available
	// for reallocation.
	
	// this is essentially "stopShow", only it's the show telling us that it stopped.
	public void notifyComplete(ShowThread showthread, Collection <DMXLightingFixture> returnedFlowers) {
		logger.info("got stop from:\t" + showthread);
		liveShows.remove(showthread);
		availableFixtures.addAll(returnedFlowers);
		logger.info("currently there are still " + liveShows.size() + " running and " + availableFixtures.size() + " fixtures unallocated");
	}

	// if any collection is complete, you'll here about it here.
	public void notifyCollectionComplete(ShowCollection collection){
		logger.info("collection " + collection.getId() + " is complete.");
		if(collection.getId().equals("hourlyShow")){
			Calendar cal = new GregorianCalendar();
			launchChimes(cal.get(Calendar.HOUR), 0, 0);
		} else if(collection.getId().equals("hourlyShowTest")){
			launchChimes(((GUI) guiWindow.gui).getChimeCount(), 0, 0);
		}
	}
	
	public Collection<DMXLightingFixture> getUnallocatedFixtures(){
		return availableFixtures;
	}
	
	public List<DMXLightingFixture> getAllFixtures(){
		return fixtures;
	}
	
	public Collection<ShowThread> getRunningShows(){
		return liveShows;
	}
	
	/**
	 * stops all shows, removes them from the pool, and reaps the fixtures.
	 */
	public void stopAll(){
		Iterator<ShowThread> i = liveShows.iterator();
		while (i.hasNext()){
			i.next().cleanStop();				
		}
		// don't need to add or remove anything from live shows here because
		// notifyComplete will do the math on the callback.			
	}

	/**
	 * Use THIS to start a show, not show.start()
	 * @param show
	 */
	public void startShow(ShowThread newshow){ // priority is so glockenspiel dosn't get trounced by singles

		// if this show is more important, steal fixtures from anyone less important.
		// (otherwise, vice versa)
		Iterator <ShowThread> i = liveShows.iterator();
		while (i.hasNext()){
			ShowThread currentShow = i.next();
			if (newshow.getShowPriority() > currentShow.getShowPriority()){
				currentShow.getFlowers().removeAll(newshow.getFlowers());
				if (currentShow.getFlowers().size() == 0){
					currentShow.cleanStop();
				}
			}else{
				newshow.getFlowers().removeAll(currentShow.getFlowers());					
			}
		}

		// if your show tried to start when there were no fixtures available
		// that you had priority on, don't bother.
		if (newshow.getFlowers().size() != 0){
			// manage show pools
			liveShows.add(newshow);
			availableFixtures.removeAll(newshow.getFlowers());

			// tell thread that we want to be notified of it's end.
			newshow.addListener(this);

			logger.info("starting:\t" + newshow);
			
			newshow.start();
		}
	}
	
	//**************************************************************************
	public void tempUpdate(float update){
		/**
		 * TODO: Used for checking temperature of installation server.
		 */
	}
	
	public void launchChimes(int hour, int minute, int second){
		if(availableFixtures.size() != 0){
			PGraphics raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);
			ShowThread newShow = null;
			for(int i=0; i<physicalColors.length; i++){
				raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
				List<DMXLightingFixture> monoFixtures = Collections.synchronizedList(new ArrayList<DMXLightingFixture>());
				Iterator <DMXLightingFixture> iter = fixtures.iterator();
				while (iter.hasNext()){
					DMXLightingFixture fixture = iter.next();
					if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals(physicalColors[i])){
						monoFixtures.add(fixture);
					}
				}
				int red = 0;
				int green = 0;
				int blue = 0;
				if(physicalColors[i].equals("red")){
					red = 255;
					green = 0;
					blue = 0;
				} else if(physicalColors[i].equals("orange")){
					red = 255;
					green = 100;
					blue = 0;
				} else if(physicalColors[i].equals("yellow")){
					red = 255;
					green = 255;
					blue = 0;
				} else if(physicalColors[i].equals("purple")){
					red = 255;
					green = 0;
					blue = 255;
				} else if(physicalColors[i].equals("pink")){
					red = 255;
					green = 150;
					blue = 150;
				}

				String soundFile = "none";
				if(i == 0){
					soundFile = systemProps.getProperty("chimeSound");
				} 
				
				newShow = new ChimesThread(monoFixtures, soundManager, 10, detectorMngr.getFps(), raster, "Chimes", ShowThread.HIGHEST, 6, 5, red, green, blue, soundFile, physicalProps);
				for(int h=1; h<hour; h++){
					newShow.chain(new ChimesThread(monoFixtures, soundManager, 10, detectorMngr.getFps(), raster, "Chimes", ShowThread.HIGHEST, 6, 5, red, green, blue, soundFile, physicalProps));
				}
				if(i < physicalColors.length-1){
					startShow(newShow);	// start every show except last one							
				}
			}
			startShow(newShow);
		}
	}
	
	public void launchGlockenspiel(int showNum, int hour, int minute, int second){
		//stopAll();
		if(availableFixtures.size() != 0){
			PGraphics raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);
			ShowThread newShow = null;
			String globalSound = null;
			String[] soundFiles = null;
			int chainDelayMin = 0;
			int chainDelayMax = 6000;
			int startDelayMin = 0;
			int startDelayMax = 6000;
			int chainCount = 4;	// default
			int soundID = soundManager.newSoundID();
			
			switch(showNum){
				case - 3:
					// hourly show TEST (westminster progression)
					ShowCollection hourlyTestCollection = new ShowCollection("hourlyShowTest");
					hourlyTestCollection.addListener(this);
					
					int quarterNote = 850;			// length of quarter note
					int wholeNote = quarterNote*4;	// 3:4 scale
					float hourlyShowGain =Float.parseFloat(systemProps.getProperty("hourShowGain"));
					String cQuarter = systemProps.getProperty("cQuarterSound");
					String dQuarter = systemProps.getProperty("dQuarterSound");
					String eQuarter = systemProps.getProperty("eQuarterSound");
					String gQuarter = systemProps.getProperty("gQuarterSound");
					String cWhole = systemProps.getProperty("cWholeSound");
					String gWhole = systemProps.getProperty("gWholeSound");
					
					newShow = new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, cQuarter, physicalProps, 0, hourlyShowGain);
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, eQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, dQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, gWhole, physicalProps, 0, hourlyShowGain));

					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, cQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, dQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, eQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, cWhole, physicalProps, 0, hourlyShowGain));
					
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, eQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, cQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, dQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, gWhole, physicalProps, 0, hourlyShowGain));
					
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, gQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, dQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, eQuarter, physicalProps, 0, hourlyShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote+quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 1, cWhole, physicalProps, 0, hourlyShowGain));	// last note fades out
								
					hourlyTestCollection.addToCollection(newShow);
					break;
				case -2:
					// hourly show (westminster progression)
					ShowCollection hourlyCollection = new ShowCollection("hourlyShow");
					hourlyCollection.addListener(this);
					
					quarterNote = 850;			// length of quarter note
					wholeNote = quarterNote*4;	// 3:4 scale
					float glockShowGain = Float.parseFloat(systemProps.getProperty("hourShowGain"));
					
					newShow = new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, "c_quarter.wav", physicalProps, 0, glockShowGain);
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, "e_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, "d_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, "g_whole.wav", physicalProps, 0, glockShowGain));

					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, "c_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, "d_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, "e_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, "c_whole.wav", physicalProps, 0, glockShowGain));
					
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, "e_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, "c_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, "d_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 2, "g_whole.wav", physicalProps, 0, glockShowGain));
					
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 0, 0, 0, "g_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 255, 0, 0, "d_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 0, 0, 255, 0, "e_quarter.wav", physicalProps, 0, glockShowGain));
					newShow.chain(new Glockenspiel(fixtures, soundManager, wholeNote+quarterNote, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, 255, 255, 255, 1, "c_whole.wav", physicalProps, 0, glockShowGain));	// last note fades out
								
					hourlyCollection.addToCollection(newShow);
					//globalSound = systemProps.getProperty("hourlyShow");
					//soundManager.globalSound(soundID,globalSound,false,1,20000,"hourlyshow");
					
					break;
				case -1:
					// light group test	(only for testing)
					newShow = new LightGroupTestThread(fixtures, soundManager, 600, detectorMngr.getFps(), raster, "LightGroupTestThread", ShowThread.HIGHEST, guiWindow.gui.loadImage("depends//images//lightgrouptest2.png"));
					break;
				case 0:		// (shows 0+ are selected randomly every 5 minutes)
					// solid color
					// create collection, and let it know we need to be notified when everyone is done.
					ShowCollection solidColorCollection = new ShowCollection("solidColor");
					solidColorCollection.addListener(this);
					float solidColorShowGain = Float.parseFloat(systemProps.getProperty("solidColorGlobalGain"));
					
					chainDelayMin = Integer.parseInt(systemProps.getProperty("solidColorGlockChainMin"));
					chainDelayMax = Integer.parseInt(systemProps.getProperty("solidColorGlockChainMax"));
					startDelayMin = Integer.parseInt(systemProps.getProperty("solidColorGlockStartMin"));
					startDelayMax = Integer.parseInt(systemProps.getProperty("solidColorGlockStartMax"));
					chainCount = Integer.parseInt(systemProps.getProperty("solidColorGlockChainCount"));
					
					soundFiles = systemProps.getProperty("solidColorShowSounds").split(",");
					Iterator <DMXLightingFixture> fixturelist = fixtures.iterator();
					while (fixturelist.hasNext()){
						DMXLightingFixture fixture = fixturelist.next();
						int red = 0;
						int green = 0;
						int blue = 0;
						if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							red = 255;
							green = 0;
							blue = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							red = 255;
							green = 100;
							blue = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							red = 255;
							green = 255;
							blue = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							red = 255;
							green = 0;
							blue = 255;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							red = 255;
							green = 150;
							blue = 150;
						}
						int bongRate = (int)(Math.random()*(chainDelayMax-chainDelayMin)) + chainDelayMin;
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
						String soundFile = soundFiles[(int)(Math.random()*(soundFiles.length-0.01))]; 	// unique per fixture, but same throughout chain
						newShow = new Glockenspiel(fixture, soundManager, 5, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, red, green, blue, 5, soundFile, physicalProps, (int)(Math.random()*(startDelayMax-startDelayMin)) + startDelayMin, solidColorShowGain);

						// add to collection
						solidColorCollection.addToCollection(newShow);
						
						for(int h=1; h<=chainCount; h++){
							newShow.chain(new Glockenspiel(fixture, soundManager, 5, detectorMngr.getFps(), raster, "Solid Color", ShowThread.HIGHEST, red, green, blue, 5, soundFile, physicalProps, bongRate, solidColorShowGain));
							bongRate = (bongRate/6) * h;	// reduces delays to create crescendo
							// entire collection must end at apex to be affective
						}
						if(fixturelist.hasNext()){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("solidColorGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,solidColorShowGain,20000,"solidcolorshow");
					
					break;
				case 1:
					// dart boards
					float[][] redlist = new float[3][3];
					redlist[0][0] = 255;	// red
					redlist[0][1] = 0;
					redlist[0][2] = 0;
					redlist[1][0] = 255;	// yellow
					redlist[1][1] = 200;
					redlist[1][2] = 0;
					redlist[2][0] = 255;	// red
					redlist[2][1] = 0;
					redlist[2][2] = 0;
					
					float[][] orangelist = new float[3][3];
					orangelist[0][0] = 255;	// orange
					orangelist[0][1] = 150;
					orangelist[0][2] = 0;
					orangelist[1][0] = 255;	// white
					orangelist[1][1] = 255;
					orangelist[1][2] = 255;
					orangelist[2][0] = 255;	// orange
					orangelist[2][1] = 150;
					orangelist[2][2] = 0;
					
					float[][] yellowlist = new float[3][3];
					yellowlist[0][0] = 0;	// dark green
					yellowlist[0][1] = 150;
					yellowlist[0][2] = 0;
					yellowlist[1][0] = 255;	// yellow
					yellowlist[1][1] = 255;
					yellowlist[1][2] = 0;	
					yellowlist[2][0] = 0;	// dark green
					yellowlist[2][1] = 150;
					yellowlist[2][2] = 0;
					
					float[][] purplelist = new float[3][3];
					purplelist[0][0] = 255;	// pink
					purplelist[0][1] = 150;
					purplelist[0][2] = 150;
					purplelist[1][0] = 0;	// blue
					purplelist[1][1] = 0;
					purplelist[1][2] = 255;
					purplelist[2][0] = 255;	// pink
					purplelist[2][1] = 150;
					purplelist[2][2] = 150;
					
					float[][] pinklist = new float[3][3];
					pinklist[0][0] = 255;	// pink
					pinklist[0][1] = 0;
					pinklist[0][2] = 100;
					pinklist[1][0] = 0;		// cyan
					pinklist[1][1] = 255;
					pinklist[1][2] = 255;
					pinklist[2][0] = 255;	// pink
					pinklist[2][1] = 0;
					pinklist[2][2] = 100;
					
					float[] pointlist = new float[3];
					pointlist[0] = 0;
					pointlist[1] = 0.5f;
					pointlist[2] = 1;

					float dartBoardGlockGain = Float.parseFloat(systemProps.getProperty("dartBoardGlockGain"));
					int dartboardDuration = Integer.parseInt(systemProps.getProperty("dartBoardGlockDuration"));
					
					// iterate through list of colors and find fixtures that match
					for(int i=0; i<physicalColors.length; i++){
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
						List<DMXLightingFixture> monoFixtures = Collections.synchronizedList(new ArrayList<DMXLightingFixture>());
						Iterator <DMXLightingFixture> iter = fixtures.iterator();
						while (iter.hasNext()){
							DMXLightingFixture fixture = iter.next();
							if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals(physicalColors[i])){
								monoFixtures.add(fixture);
							}
						}
						float[][] colorlist = new float[0][0];
						if(physicalColors[i].equals("red")){
							colorlist = redlist;
						} else if(physicalColors[i].equals("orange")){
							colorlist = orangelist;
						} else if(physicalColors[i].equals("yellow")){
							colorlist = yellowlist;
						} else if(physicalColors[i].equals("purple")){
							colorlist = purplelist;
						} else if(physicalColors[i].equals("pink")){
							colorlist = pinklist;
						}
						ColorScheme spectrum = new ColorScheme(colorlist, pointlist);
						float throbspeed =  (float)(Math.random()*0.001f)+0.0005f;
						newShow = new DartBoardThread(monoFixtures, soundManager, dartboardDuration, detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.HIGHEST, spectrum, (float)(Math.random()*0.02f)+0.01f, 0.1f, throbspeed, throbspeed, 0.12f, "none", physicalProps, false, guiWindow.gui.loadImage("depends//images//sprites//randomwhitespots.png"), dartBoardGlockGain);
						if(i < physicalColors.length-1){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("dartBoardGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,dartBoardGlockGain,20000,"dartboardshow");
					
					break;
				case 2:
					// sparkle spiral
					
					soundFiles = systemProps.getProperty("sparkleSpiralShowSounds").split(",");
					Iterator <DMXLightingFixture> fiter = fixtures.iterator();
					float sparkleSpiralShowGain = Float.parseFloat(systemProps.getProperty("sparkleSpiralShowGain"));
					chainDelayMin = Integer.parseInt(systemProps.getProperty("sparkleSpiralGlockChainMin"));
					chainDelayMax = Integer.parseInt(systemProps.getProperty("sparkleSpiralGlockChainMax"));
					startDelayMin = Integer.parseInt(systemProps.getProperty("sparkleSpiralGlockStartMin"));
					startDelayMax = Integer.parseInt(systemProps.getProperty("sparkleSpiralGlockStartMax"));
					chainCount = Integer.parseInt(systemProps.getProperty("sparkleSpiralGlockChainCount"));
					
					while (fiter.hasNext()){
						DMXLightingFixture fixture = fiter.next();
						
						float[] points = new float[2];
						points[0] = 0;
						points[1] = 1;
						float[][] colorlist = new float[2][3];
						if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							colorlist[0][0] = 255;	// red
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;	// white
							colorlist[1][1] = 255;
							colorlist[1][2] = 255;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							colorlist[0][0] = 255;	// orange
							colorlist[0][1] = 150;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;	// white
							colorlist[1][1] = 255;
							colorlist[1][2] = 255;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							colorlist[0][0] = 255;	// yellow
							colorlist[0][1] = 255;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;	// white
							colorlist[1][1] = 255;
							colorlist[1][2] = 255;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							colorlist[0][0] = 255;	// white
							colorlist[0][1] = 255;
							colorlist[0][2] = 255;
							colorlist[1][0] = 0;	// black
							colorlist[1][1] = 0;
							colorlist[1][2] = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							colorlist[0][0] = 255;	// purple
							colorlist[0][1] = 0;
							colorlist[0][2] = 255;
							colorlist[1][0] = 255;	// white
							colorlist[1][1] = 255;
							colorlist[1][2] = 255;
						}
						
						ColorScheme spectrum = new ColorScheme(colorlist, points);
						int bongRate = (int)(Math.random()*(chainDelayMax-chainDelayMin)) + chainDelayMin;
		            	String soundFile = soundFiles[(int)(Math.random()*(soundFiles.length-0.01))]; 	// unique per fixture, but same throughout chain
						raster = guiWindow.gui.createGraphics(fixture.getWidth(), fixture.getHeight(), PConstants.P3D);	// needs a unique raster for each color
						newShow = new SparkleSpiralThread(fixture, soundManager, 20, detectorMngr.getFps(), raster, "Sparkle Spiral", ShowThread.HIGHEST, spectrum, 0, 0, false, soundFile, physicalProps, (int)(Math.random()*(startDelayMax-startDelayMin)) + startDelayMin, sparkleSpiralShowGain);
						for(int h=1; h<=chainCount; h++){
							newShow.chain(new SparkleSpiralThread(fixture, soundManager, 20, detectorMngr.getFps(), raster, "Sparkle Spiral", ShowThread.HIGHEST, spectrum, 0, 0, false, soundFile, physicalProps, bongRate, sparkleSpiralShowGain));
						}
						if(fiter.hasNext()){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("sparkleSpiralGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,sparkleSpiralShowGain,20000,"sparklespiralshow");
					
					break;
				case 3:
					// gradient rings
					int gradientringsDuration = Integer.parseInt(systemProps.getProperty("gradientRingsGlockDuration"));
					Iterator <DMXLightingFixture> fixturearray = fixtures.iterator();
					float gradientRingsGlobalGain = Float.parseFloat(systemProps.getProperty("gradientRingsGlobalGain"));
					
					while (fixturearray.hasNext()){
						DMXLightingFixture fixture = fixturearray.next();
						if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							innerRing = innerRingRed;
							outerRing = outerRingRed;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							innerRing = innerRingOrange;
							outerRing = outerRingOrange;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							innerRing = innerRingYellow;
							outerRing = outerRingYellow;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							innerRing = innerRingPink;
							outerRing = outerRingPink;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							innerRing = innerRingPurple;
							outerRing = outerRingPurple;
						}
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
						newShow = new SpinningRingThread(fixture, soundManager, gradientringsDuration, detectorMngr.getFps(), raster, "GradientRings", ShowThread.HIGHEST, 255, 255, 255, (int)(Math.random()*5), (int)(Math.random()*10), 20, 2, 0.05f, 0.05f, 0.05f, 0.05f, outerRing, innerRing, false, "none", physicalProps, (int)(Math.random()*6000), false, gradientRingsGlobalGain);
						//newShow = new SpinningThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "Sweep", ShowThread.LOW, sweepSprite, 2, 0.1f, 0.2f, 5, "none", physicalProps, (int)(Math.random()*6000));
						if(fixturearray.hasNext()){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("gradientRingsGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,gradientRingsGlobalGain,20000,"gradientringsshow");
					
					break;
				case 4:
					// vertical color shift
					int verticalColorDuration = Integer.parseInt(systemProps.getProperty("verticalColorGlockDuration"));
					for(int i=0; i<floors.length; i++){
						// iterate through list of floors and find fixtures that match
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each set
						List<DMXLightingFixture> fixtureset = Collections.synchronizedList(new ArrayList<DMXLightingFixture>());
						Iterator <DMXLightingFixture> iter = fixtures.iterator();
						while (iter.hasNext()){
							DMXLightingFixture fixture = iter.next();
							if(physicalProps.getProperty(fixture.getID()).split(",")[2].equals(floors[i])){
								fixtureset.add(fixture);
							}
						}
						
						float[] points = new float[5];
						points[0] = 0;
						points[1] = 0.25f;
						points[2] = 0.5f;
						points[3] = 0.75f;
						points[4] = 1;
						float[][] colorlist = new float[5][3];
						colorlist[0][0] = 0;	// cyan
						colorlist[0][1] = 255;
						colorlist[0][2] = 255;
						colorlist[1][0] = 255;	// yellow
						colorlist[1][1] = 255;
						colorlist[1][2] = 0;
						colorlist[2][0] = 255;	// red
						colorlist[2][1] = 0;
						colorlist[2][2] = 0;
						colorlist[3][0] = 200;	// violet
						colorlist[3][1] = 75;
						colorlist[3][2] = 255;
						colorlist[4][0] = 0;	// cyan
						colorlist[4][1] = 255;
						colorlist[4][2] = 255;
						
						
						ColorScheme spectrum = new ColorScheme(colorlist, points);
						newShow = new GlobalColorShift(fixtureset, soundManager, verticalColorDuration, detectorMngr.getFps(), raster, "VerticalColorShift", ShowThread.HIGHEST, spectrum, 0.005f, i*0.1f, "Piano_Ballad.wav", physicalProps);
						if(i < floors.length-1){
							startShow(newShow);	// start every show except last one							
						}
					}

					float verticalColorShiftGlobalGain = Float.parseFloat(systemProps.getProperty("verticalColorShiftGlobalGain"));
					globalSound = systemProps.getProperty("verticalColorShiftGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,verticalColorShiftGlobalGain,20000,"verticalcolorshiftshow");
					
					break;
				case 5:
					// radial color shift
					int radialColorDuration = Integer.parseInt(systemProps.getProperty("radialColorGlockDuration"));
					for(int i=0; i<sectors.length; i++){
						// iterate through list of sectors and find fixtures that match
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each set
						List<DMXLightingFixture> fixtureset = Collections.synchronizedList(new ArrayList<DMXLightingFixture>());
						Iterator <DMXLightingFixture> iter = fixtures.iterator();
						while (iter.hasNext()){
							DMXLightingFixture fixture = iter.next();
							if(physicalProps.getProperty(fixture.getID()).split(",")[3].equals(String.valueOf(i))){
								fixtureset.add(fixture);
							}
						}
						
						/*
						// grayscale
						float[] points = new float[3];
						points[0] = 0;
						points[1] = 0.05f;
						points[2] = 1;
						float[][] colorlist = new float[3][3];
						colorlist[0][0] = 255;
						colorlist[0][1] = 255;
						colorlist[0][2] = 255;
						colorlist[1][0] = 0;
						colorlist[1][1] = 0;
						colorlist[1][2] = 0;
						colorlist[2][0] = 0;
						colorlist[2][1] = 0;
						colorlist[2][2] = 0;
						*/
						
						float[] points = new float[5];
						points[0] = 0;
						points[1] = 0.25f;
						points[2] = 0.5f;
						points[3] = 0.75f;
						points[4] = 1;
						
						float[][] colorlist = new float[5][3];
						colorlist[0][0] = 0;	// cyan
						colorlist[0][1] = 255;
						colorlist[0][2] = 255;
						colorlist[1][0] = 255;	// white
						colorlist[1][1] = 255;
						colorlist[1][2] = 255;
						colorlist[2][0] = 255;	// magenta
						colorlist[2][1] = 0;
						colorlist[2][2] = 150;
						colorlist[3][0] = 255;	// white
						colorlist[3][1] = 255;
						colorlist[3][2] = 255;
						colorlist[4][0] = 0;	// cyan
						colorlist[4][1] = 255;
						colorlist[4][2] = 255;
						/*
						float[][] colorlist = new float[5][3];
						colorlist[0][0] = 0;	// cyan
						colorlist[0][1] = 255;
						colorlist[0][2] = 255;
						colorlist[1][0] = 255;	// yellow
						colorlist[1][1] = 255;
						colorlist[1][2] = 0;
						colorlist[2][0] = 255;	// red
						colorlist[2][1] = 0;
						colorlist[2][2] = 0;
						colorlist[3][0] = 200;	// violet
						colorlist[3][1] = 75;
						colorlist[3][2] = 255;
						colorlist[4][0] = 0;	// cyan
						colorlist[4][1] = 255;
						colorlist[4][2] = 255;
						*/
						
						ColorScheme spectrum = new ColorScheme(colorlist, points);
						newShow = new GlobalColorShift(fixtureset, soundManager, radialColorDuration, detectorMngr.getFps(), raster, "RadialColorShift", ShowThread.HIGHEST, spectrum, 0.02f, i/(float)sectors.length, "none", physicalProps);
						if(i < sectors.length-1){
							startShow(newShow);	// start every show except last one							
						}
					}

					float radialColorShiftGlobalGain = Float.parseFloat(systemProps.getProperty("radialColorShiftGlobalGain"));
					globalSound = systemProps.getProperty("radialColorShiftGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,radialColorShiftGlobalGain,20000,"radialcolorshiftshow");
					
					break;
				case 6:
					// fireworks
					soundFiles = systemProps.getProperty("fireworksShowSounds").split(",");
					Iterator <DMXLightingFixture> floweriter = fixtures.iterator();
					float fireworksGlobalGain = Float.parseFloat(systemProps.getProperty("fireworksGlobalGain"));
					chainDelayMin = Integer.parseInt(systemProps.getProperty("fireworksGlockChainMin"));
					chainDelayMax = Integer.parseInt(systemProps.getProperty("fireworksGlockChainMax"));
					startDelayMin = Integer.parseInt(systemProps.getProperty("fireworksGlockStartMin"));
					startDelayMax = Integer.parseInt(systemProps.getProperty("fireworksGlockStartMax"));
					chainCount = Integer.parseInt(systemProps.getProperty("fireworksGlockChainCount"));
					
					while (floweriter.hasNext()){
						DMXLightingFixture fixture = floweriter.next();
						
						float[] points = new float[2];
						points[0] = 0;
						points[1] = 1;
						float[][] colorlist = new float[2][3];
						if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("red")){
							colorlist[0][0] = 255;	// red
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;	// yellow
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("orange")){
							colorlist[0][0] = 255;	// orange
							colorlist[0][1] = 100;
							colorlist[0][2] = 0;
							colorlist[1][0] = 255;	// white
							colorlist[1][1] = 255;
							colorlist[1][2] = 255;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("yellow")){
							colorlist[0][0] = 255;	// yellow
							colorlist[0][1] = 255;
							colorlist[0][2] = 0;
							colorlist[1][0] = 0;	// green
							colorlist[1][1] = 255;
							colorlist[1][2] = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("pink")){
							colorlist[0][0] = 255;	// white
							colorlist[0][1] = 255;
							colorlist[0][2] = 255;
							colorlist[1][0] = 255;	// red
							colorlist[1][1] = 0;
							colorlist[1][2] = 0;
						} else if(physicalProps.getProperty(fixture.getID()).split(",")[0].equals("purple")){
							colorlist[0][0] = 255;	// red
							colorlist[0][1] = 0;
							colorlist[0][2] = 0;
							colorlist[1][0] = 0;	// blue
							colorlist[1][1] = 0;
							colorlist[1][2] = 255;
						}
						
						ColorScheme spectrum = new ColorScheme(colorlist, points);
						//int bongRate = (int)(Math.random()*(chainDelayMax-chainDelayMin)) + chainDelayMin;
		            	int duration = (int)(Math.random()*5) + 5;
		            	String soundFile = soundFiles[(int)(Math.random()*(soundFiles.length-0.01))]; 	// unique per fixture, but same throughout chain
		            	raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
						newShow = new FireworksThread(fixture, soundManager, duration, detectorMngr.getFps(), raster, "FireworksThread", ShowThread.HIGHEST, spectrum, 8, 0.8f, guiWindow.gui.loadImage("depends//images//sprites//thickring.png"), soundFile, physicalProps, (int)(Math.random()*(startDelayMax-startDelayMin)) + startDelayMin, false, fireworksGlobalGain);
						for(int h=1; h<=chainCount; h++){
							newShow.chain(new FireworksThread(fixture, soundManager, duration, detectorMngr.getFps(), raster, "FireworksThread", ShowThread.HIGHEST, spectrum, 8, 0.8f, guiWindow.gui.loadImage("depends//images//sprites//thickring.png"), soundFile, physicalProps, (int)(Math.random()*(chainDelayMax-chainDelayMin)) + chainDelayMin, false, fireworksGlobalGain));
						}
						if(floweriter.hasNext()){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("fireworksGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,fireworksGlobalGain,20000,"fireworksshow");
					
					break;
				case 7:
					// sweeps
					Iterator <DMXLightingFixture> fixtureiter = fixtures.iterator();
					int luckynumber = (int)(Math.random()*(sweepSpriteList.length-1));
					float sweepsGlobalGain = Float.parseFloat(systemProps.getProperty("sweepsGlobalGain"));
					
					while (fixtureiter.hasNext()){
						DMXLightingFixture fixture = fixtureiter.next();
						sweepSprite = sweepSpriteList[luckynumber];
						raster = guiWindow.gui.createGraphics(fixtures.get(0).getWidth(), fixtures.get(0).getHeight(), PConstants.P3D);	// needs a unique raster for each color
						newShow = new SpinningThread(fixture, soundManager, 30, detectorMngr.getFps(), raster, "Sweep", ShowThread.HIGHEST, sweepSprite, 2, 0.1f, 0.2f, 5, "none", physicalProps, 0, false, sweepsGlobalGain);
						luckynumber++;
						if(luckynumber == sweepSpriteList.length){
							luckynumber = 0;
						}
						if(fixtureiter.hasNext()){
							startShow(newShow);	// start every show except last one							
						}
					}
					
					globalSound = systemProps.getProperty("sweepsGlobalSound");
					soundManager.globalSound(soundID,globalSound,false,sweepsGlobalGain,20000,"sweepsshow");
					
					break;
			}
			startShow(newShow);
		}
	}
	
	public void timedEvent(TimedEvent e){
		//System.out.println(e.hour+":"+e.minute+":"+e.sec);
		if(e.hour > 12){
			if(e.minute == 0){
				//launchChimes(e.hour-12, e.minute, e.sec);
				launchGlockenspiel((int)-2, e.hour-12, e.minute, e.sec);
			} else {
				launchGlockenspiel((int)((timedShows.length-0.01)*Math.random()), e.hour-12, e.minute, e.sec);
			}
		} else {
			if(e.minute == 0){
				//launchChimes(e.hour, e.minute, e.sec);
				launchGlockenspiel((int)-2, e.hour, e.minute, e.sec);
			} else {
				launchGlockenspiel((int)((timedShows.length-0.01)*Math.random()), e.hour, e.minute, e.sec);
			}
		}
	}
	
	public void weatherChanged(WeatherChangedEvent wce){
		if(wce.hasSunriseChanged()) {
			Calendar sunrise = wce.getRecord().getSunrise();
			int h = sunrise.get(Calendar.HOUR_OF_DAY);
			int m = sunrise.get(Calendar.MINUTE);
			int s = sunrise.get(Calendar.SECOND);
			logger.info("Sunrise at " + h + ":" + m + ":" + s);
			//sunriseOn.reschedule(h-1, m, s); // turn off an hour before sunrise
		}
		if(wce.hasSunsetChanged()) {
			Calendar sunset = wce.getRecord().getSunset();
			int h = sunset.get(Calendar.HOUR_OF_DAY);
			int m = sunset.get(Calendar.MINUTE);
			int s = sunset.get(Calendar.SECOND);
			logger.info("Sunset at " + h + ":" + m + ":" + s);
			//sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
		}

		logger.debug("CONDITION = " + wce.getRecord().getCondition());
		logger.debug("VISIBILITY = " + wce.getRecord().getVisibility());
		logger.debug("OUTSIDE TEMP = " + wce.getRecord().getOutsideTemperature());
	}
	
	public static void main(String[] args) {					// PROGRAM LAUNCH
		new Conductor(args);
	}


}