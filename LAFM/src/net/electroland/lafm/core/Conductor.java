package net.electroland.lafm.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import org.apache.log4j.Logger;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;
import net.electroland.lafm.gui.GUIWindow;
import net.electroland.lafm.scheduler.TimedEvent;
import net.electroland.lafm.scheduler.TimedEventListener;
import net.electroland.lafm.shows.AdditivePropellerThread;
import net.electroland.lafm.shows.ChimesThread;
import net.electroland.lafm.shows.DartBoardThread;
import net.electroland.lafm.shows.FireworksThread;
import net.electroland.lafm.shows.Glockenspiel;
import net.electroland.lafm.shows.ImageSequenceThread;
import net.electroland.lafm.shows.LightGroupTestThread;
import net.electroland.lafm.shows.PieThread;
import net.electroland.lafm.shows.PropellerThread;
import net.electroland.lafm.shows.ShutdownThread;
import net.electroland.lafm.shows.SpinningRingThread;
import net.electroland.lafm.shows.SpiralThread;
import net.electroland.lafm.shows.ThrobbingThread;
import net.electroland.lafm.shows.VegasThread;
import net.electroland.lafm.weather.WeatherChangeListener;
import net.electroland.lafm.weather.WeatherChangedEvent;
import net.electroland.lafm.weather.WeatherChecker;
import net.electroland.lafm.util.ColorScheme;
import processing.core.PConstants;
import processing.core.PGraphics;
import promidi.MidiIO;
import promidi.Note;

public class Conductor extends Thread implements ShowThreadListener, WeatherChangeListener, TimedEventListener{

	static Logger logger = Logger.getLogger(Conductor.class);
	
	public GUIWindow guiWindow;				// frame for GUI
	//static public DMXLightingFixture[] flowers;		// all flower fixtures
	public MidiIO midiIO;						// sensor data IO
	public DetectorManager detectorMngr;
	public SoundManager soundManager;
	public WeatherChecker weatherChecker;
	public Properties sensors;					// pitch to fixture mappings
	public Properties systemProps;
	public TimedEvent[] clockEvents;
	private ImageSequenceCache imageCache; 	// for ImageSequenceThreads
	public String[] sensorShows, timedShows;	// list of names of sensor-triggered shows
	public String[] fixtureActivity;			// 22 fixtures, null if empty; show name if in use
	public int currentSensorShow;				// number of show to display when sensor is triggered
	//private int hitCount = 0;					// increments each time sensor is triggered
	//private int hitCountMax = 20;				// number of hits before switching sensor triggered show
	public boolean forceSensorShow = false;

	// sample timed events, but I assume the building will be closed for some time at night
	//TimedEvent sunriseOn = new TimedEvent(6,00,00, this); // on at sunrise-1 based on weather
	//TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather

	private List <ShowThread>liveShows;
	private List <DMXLightingFixture> availableFixtures;
	private List <DMXLightingFixture> fixtures;

	public Conductor(String args[]){
		
		try{
			systemProps = new Properties();
			systemProps.load(new FileInputStream(new File("depends//system.properties")));
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
		
		//hitCountMax = Integer.parseInt(systemProps.getProperty("hitCountThreshold"));

		// to track which fixtures are used, and what shows are currently running.
		liveShows = Collections.synchronizedList(new ArrayList<ShowThread>());
		availableFixtures = Collections.synchronizedList(new ArrayList<DMXLightingFixture>(detectorMngr.getFixtures()));
		
		fixtureActivity = new String[22];	// all null to begin with
		
		currentSensorShow = 0;
		sensorShows = new String[16];	// size dependent on number of sensor-triggered shows
		sensorShows[0] = "Throb";
		sensorShows[1] = "Propeller";
		sensorShows[2] = "Spiral";
		sensorShows[3] = "Dart Board";
		sensorShows[4] = "Pie";
		sensorShows[5] = "Bubbles";
		sensorShows[6] = "Matrix Rings";
		sensorShows[7] = "Flashing Pie";
		sensorShows[8] = "Vegas";
		sensorShows[9] = "Fireworks";
		sensorShows[10] = "Additive Propeller";
		sensorShows[11] = "bees";
		sensorShows[12] = "explode";
		sensorShows[13] = "swirlPulse";
		sensorShows[14] = "Spinning Rings";
		sensorShows[15] = "Light Group Test";
		
		timedShows = new String[3];
		timedShows[0] = "Solid Color";
		timedShows[1] = "Light Group Test";
		timedShows[2] = "Chimes";
		
		sensors = new Properties();
		try{

			// load sensor info
			sensors.load(new FileInputStream(new File("depends//sensors.properties")));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		clockEvents = new TimedEvent[24*12];	// event every 5 minutes just for testing
		for(int h=0; h<24; h++){
			for(int m=0; m<12; m++){
				clockEvents[(h+1)*m] = new TimedEvent(h,m*5,0,this);
			}
		}

		midiIO = MidiIO.getInstance();
		try{
			midiIO.plug(this, "midiEvent", Integer.parseInt(systemProps.getProperty("midiDeviceNumber")), 0);	// device # and midi channel
		} catch(Exception e){
			e.printStackTrace();
		}
		soundManager = new SoundManager(systemProps.getProperty("soundAddress"), Integer.parseInt(systemProps.getProperty("soundPort")));
		guiWindow = new GUIWindow(this, detectorMngr.getDetectors());
		guiWindow.setVisible(true);
		
		try {
			Properties imageProps = new Properties();
			imageProps.load(new FileInputStream(new File("depends//images.properties")));
			imageCache = new ImageSequenceCache(imageProps, guiWindow.gui);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
		//weatherChecker.start();

		Runtime.getRuntime().addShutdownHook(new ShutdownThread(fixtures, guiWindow.gui.createGraphics(256, 256, PConstants.P2D), "ShutdownShow"));
	
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
				//PGraphics2D raster = new PGraphics2D(256,256,null);
				PGraphics raster = guiWindow.gui.createGraphics(256, 256, PConstants.P3D);
				ShowThread newShow = new ThrobbingThread(fixture, null, 5, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 500, 500, 0, 0, 0, 0);	// default
				
				if(forceSensorShow){
					
					// THIS IS ONLY RUN WHEN DEFAULT SENSOR SHOWS ARE OVERRIDDEN BY GUI SELECTION

					/*
					// changes sensor triggers based on hits (not used)
					if(hitCount < hitCountMax){
						hitCount++;
					} else {
						hitCount = 0;
						if(currentSensorShow < sensorShows.length){
							currentSensorShow++;
						} else {
							currentSensorShow = 0;
						}
					}
					*/
					
					float[][] colorlist;
					float[] pointlist;
					ColorScheme spectrum;
					
					switch (currentSensorShow) {
			            case 0:
			            	newShow = new ThrobbingThread(fixture, null, 5, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 250, 250, 0, 0, 0, 0);
			            	break;
			            case 1:
			            	newShow = new PropellerThread(fixture, null, 5, detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, 255, 0, 0, 20, 5, 0.1f, 0.1f);
			            	break;
			            case 2:
			            	newShow = new SpiralThread(fixture, null, 10, detectorMngr.getFps(), raster, "SpiralThread", ShowThread.LOW, 0, 255, 255, 30, 2, 3, 100, guiWindow.gui.loadImage("depends//images//sprites//sphere50alpha.png"));
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
							newShow = new DartBoardThread(fixture, null, 20, detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.LOW, spectrum, 0.01f, 0.1f, 0.001f, 0.002f);
							break;
			            case 4:
			            	newShow = new PieThread(fixture, null, 3, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255, 255, 0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"));
			            	break;
			            case 5:
			            	newShow = new ImageSequenceThread(fixture, null, 5, detectorMngr.getFps(), raster, "BubblesThread", ShowThread.LOW, imageCache.getSequence("bubbles"), false);					
			            	break;
			            case 6:
			            	newShow = new ImageSequenceThread(fixture, null, 5, detectorMngr.getFps(), raster, "MatrixRingsThread", ShowThread.LOW, imageCache.getSequence("matrixRings"), false);					
							((ImageSequenceThread)newShow).enableTint(90, 100);
							break;
			            case 7: 
			            	newShow = new PieThread(fixture, null, 2, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255,255,0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"));
							newShow.chain(new ThrobbingThread(fixture, null, 1, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255,255,0, 100, 100, 0, 0, 0, 0));									
							break;
			            case 8:
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
							newShow = new VegasThread(fixture, null, 5, detectorMngr.getFps(), raster, "VegasThread", ShowThread.LOW, spectrum, 0);
							//newShow = new ExpandingThread(fixture, null,2, detectorMngr.getFps(), raster, "ExpandingThread", ShowThread.LOW, guiWindow.gui.loadImage("depends//images//sprites//sphere50alpha.png"));
							//newShow.chain(new PieThread(fixture, null, 2, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255, 255, 0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png")));
							//newShow.chain(new ImageSequenceThread(fixture, null, 2, detectorMngr.getFps(), raster, "BubblesThread", ShowThread.LOW, imageCache.getSequence("redThrob"), false));		
							break;
			            case 9:
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
							newShow = new FireworksThread(fixture, null, 5, detectorMngr.getFps(), raster, "FireworksThread", ShowThread.LOW, spectrum, 8, 0.8f, guiWindow.gui.loadImage("depends//images//sprites//ring50alpha.png"));
							break;
			            case 10: 
			            	newShow = new AdditivePropellerThread(fixture, null, 30, detectorMngr.getFps(), raster, "AdditivePropellerThread", ShowThread.LOW, 0.2f, 5, 0.1f, 0.1f, guiWindow.gui.createGraphics(256, 256, PConstants.P3D), guiWindow.gui.createGraphics(256, 256, PConstants.P3D), guiWindow.gui.createGraphics(256, 256, PConstants.P3D));
			            	break;
			            case 11:
			            	// bees
			            	newShow = new ImageSequenceThread(fixture, null, 5, detectorMngr.getFps(), raster, "BeesThread", ShowThread.LOW, imageCache.getSequence("bees"), false);					
							((ImageSequenceThread)newShow).enableTint(90, 100);
							break;
			            case 12:
			            	// explode
			            	newShow = new ImageSequenceThread(fixture, null, 5, detectorMngr.getFps(), raster, "ExplodeThread", ShowThread.LOW, imageCache.getSequence("explode"), false);					
							((ImageSequenceThread)newShow).enableTint(90, 100);
							break;
			            case 13:
			            	//swirlPulse
			            	newShow = new ImageSequenceThread(fixture, null, 5, detectorMngr.getFps(), raster, "SwirlPulseThread", ShowThread.LOW, imageCache.getSequence("swirlPulse"), false);					
							((ImageSequenceThread)newShow).enableTint(90, 100);
							break;
			            case 14:
			            	// spinning rings
			            	newShow = new SpinningRingThread(fixture, null, 30, detectorMngr.getFps(), raster, "SpinningRingThread", ShowThread.LOW, 255, 0, 0, 0.01f, 2, 20, 5, 0.05f, 0.05f, guiWindow.gui.loadImage("depends//images//sprites//dashedring256alpha.png"), guiWindow.gui.loadImage("depends//images//sprites//dashedring152alpha.png"));
			            	break;
			            case 15:
			            	// light group test
			            	newShow = new LightGroupTestThread(fixture, null, 30, detectorMngr.getFps(), raster, "SpinningRingThread", ShowThread.LOW, guiWindow.gui.loadImage("depends//images//lightgrouptest.png"));
					}
					
					
				} else {
					String[] showProps = systemProps.getProperty(fixtureId).split(",");
					
					if(showProps[0].equals("propeller")){
						newShow = new PropellerThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), Float.parseFloat(showProps[7]), Float.parseFloat(showProps[8]));
					} else if(showProps[0].equals("throb")){
						newShow = new ThrobbingThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), Integer.parseInt(showProps[7]), Integer.parseInt(showProps[8]), Integer.parseInt(showProps[9]), Integer.parseInt(showProps[10]));
					} else if(showProps[0].equals("spiral")){
						newShow = new SpiralThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "SpiralThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), Integer.parseInt(showProps[5]), Integer.parseInt(showProps[6]), Integer.parseInt(showProps[7]), Integer.parseInt(showProps[8]), guiWindow.gui.loadImage("depends//images//sprites//sphere50alpha.png"));
					} else if(showProps[0].equals("dartboard")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new DartBoardThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), Float.parseFloat(showProps[6]), Float.parseFloat(showProps[7]));
					} else if(showProps[0].equals("images")){
						newShow = new ImageSequenceThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, showProps[2], ShowThread.LOW, imageCache.getSequence(showProps[2]), false);					
						((ImageSequenceThread)newShow).enableTint(Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]));
					} else if(showProps[0].equals("vegas")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new VegasThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "VegasThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]));
					} else if(showProps[0].equals("fireworks")){
						ColorScheme spectrum = processColorScheme(showProps[2], showProps[3]);
						newShow = new FireworksThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "FireworksThread", ShowThread.LOW, spectrum, Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), guiWindow.gui.loadImage("depends//images//sprites//ring50alpha.png"));
					} else if(showProps[0].equals("additivepropeller")){
						newShow = new AdditivePropellerThread(fixture, null, Integer.parseInt(showProps[1]), detectorMngr.getFps(), raster, "AdditivePropellerThread", ShowThread.LOW, Float.parseFloat(showProps[2]), Integer.parseInt(showProps[3]), Float.parseFloat(showProps[4]), Float.parseFloat(showProps[5]), guiWindow.gui.createGraphics(256, 256, PConstants.P3D), guiWindow.gui.createGraphics(256, 256, PConstants.P3D), guiWindow.gui.createGraphics(256, 256, PConstants.P3D));
					} else if(showProps[0].equals("flashingpie")){
						newShow = new PieThread(fixture, null, Integer.parseInt(showProps[5]), detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"));
						newShow.chain(new ThrobbingThread(fixture, null, Integer.parseInt(showProps[6]), detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, Integer.parseInt(showProps[2]), Integer.parseInt(showProps[3]), Integer.parseInt(showProps[4]), 300, 300, 0, 0, 0, 0));									
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

	public Collection<DMXLightingFixture> getUnallocatedFixtures(){
		return availableFixtures;
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
	
	public void launchGlockenspiel(int showNum){
		//stopAll();
		if(availableFixtures.size() != 0){
			PGraphics raster = guiWindow.gui.createGraphics(256, 256, PConstants.P3D);
			ShowThread newShow = null;
			switch(showNum){
				case 0:
					newShow = new Glockenspiel(fixtures, soundManager, 5, detectorMngr.getFps(), raster, "Glockenspiel", ShowThread.HIGHEST, 0, 30, 0);
					break;
				case 1:
					newShow = new LightGroupTestThread(fixtures, null, 30, detectorMngr.getFps(), raster, "LightGroupTestThread", ShowThread.HIGHEST, guiWindow.gui.loadImage("depends//images//lightgrouptest.png"));
					break;
				case 2:
					newShow = new ChimesThread(fixtures, soundManager, 60, detectorMngr.getFps(), raster, "Chimes", ShowThread.HIGHEST, 6, 5, 0, 255, 255);
					break;
			}
			startShow(newShow);
		}
	}
	
	public void timedEvent(TimedEvent e){
		//System.out.println(e.hour+":"+e.minute+":"+e.sec);
		//PGraphics2D raster = new PGraphics2D(256,256,null);
		PGraphics raster = guiWindow.gui.createGraphics(256, 256, PConstants.P3D);
		ShowThread newShow = new Glockenspiel(fixtures, soundManager, 5, detectorMngr.getFps(), raster, "Glockenspiel", ShowThread.HIGHEST, e.hour, e.minute, e.sec);
		startShow(newShow);
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