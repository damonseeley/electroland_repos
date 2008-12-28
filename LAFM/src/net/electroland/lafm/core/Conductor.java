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

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;
import net.electroland.lafm.gui.GUI;
import net.electroland.lafm.gui.GUIWindow;
import net.electroland.lafm.scheduler.TimedEvent;
import net.electroland.lafm.scheduler.TimedEventListener;
import net.electroland.lafm.shows.DartBoardThread;
import net.electroland.lafm.shows.Glockenspiel;
import net.electroland.lafm.shows.ImageSequenceThread;
import net.electroland.lafm.shows.PieThread;
import net.electroland.lafm.shows.PropellerThread;
import net.electroland.lafm.shows.ShutdownThread;
import net.electroland.lafm.shows.SpiralThread;
import net.electroland.lafm.shows.ThrobbingThread;
import net.electroland.lafm.weather.WeatherChangeListener;
import net.electroland.lafm.weather.WeatherChangedEvent;
import net.electroland.lafm.weather.WeatherChecker;
import net.electroland.lafm.util.ColorScheme;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PGraphics2D;
import promidi.MidiIO;
import promidi.Note;

public class Conductor extends Thread implements ShowThreadListener, WeatherChangeListener, TimedEventListener{
	
	public GUIWindow guiWindow;				// frame for GUI
	//static public DMXLightingFixture[] flowers;		// all flower fixtures
	public MidiIO midiIO;						// sensor data IO
	public DetectorManager detectorMngr;
	public SoundManager soundManager;
	public WeatherChecker weatherChecker;
	public Properties sensors;					// pitch to fixture mappings
	public TimedEvent[] clockEvents;
	private ImageSequenceCache imageCache; 	// for ImageSequenceThreads
	public String[] sensorShows;				// list of names of sensor-triggered shows
	public String[] fixtureActivity;			// 22 fixtures, null if empty; show name if in use
	public int currentSensorShow;				// number of show to display when sensor is triggered

	// sample timed events, but I assume the building will be closed for some time at night
	//TimedEvent sunriseOn = new TimedEvent(6,00,00, this); // on at sunrise-1 based on weather
	//TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather

	private List <ShowThread>liveShows;
	private List <DMXLightingFixture> availableFixtures;
	private List <DMXLightingFixture> fixtures;

	public Conductor(String args[]){
			
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

		// to track which fixtures are used, and what shows are currently running.
		liveShows = Collections.synchronizedList(new ArrayList<ShowThread>());
		availableFixtures = Collections.synchronizedList(new ArrayList<DMXLightingFixture>(detectorMngr.getFixtures()));
		
		fixtureActivity = new String[22];	// all null to begin with
		
		currentSensorShow = 0;
		sensorShows = new String[6];	// size dependent on number of sensor-triggered shows
		sensorShows[0] = "Image Sequence";
		sensorShows[1] = "Throb";
		sensorShows[2] = "Propeller";
		sensorShows[3] = "Spiral";
		sensorShows[4] = "Dart Board";
		sensorShows[5] = "Pie";
		
		sensors = new Properties();
		try{

			// load sensor info
			sensors.load(new FileInputStream(new File("depends//sensors.properties")));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		clockEvents = new TimedEvent[24*60];	// event every minute just for testing
		for(int h=0; h<24; h++){
			for(int m=0; m<60; m++){
				clockEvents[(h+1)*m] = new TimedEvent(h,m,0,this);
			}
		}

		midiIO = MidiIO.getInstance();
		try{
			midiIO.plug(this, "midiEvent", 0, 0);	// device # and midi channel
		} catch(Exception e){
			e.printStackTrace();
		}
		soundManager = new SoundManager();
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
				ShowThread newShow = new ThrobbingThread(fixture, null, 60, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 500, 500, 0, 0);	// default
				if (note.getPitch() == 36){
					//newShow = new DiagnosticThread(fixture, null, 60, detectorMngr.getFps(), raster, "DiagnosticThread");
					newShow = new ImageSequenceThread(fixture, null, 60, detectorMngr.getFps(), raster, "ImageSequenceThread", ShowThread.LOW, imageCache.getSequence("redThrob"), false);					
					((ImageSequenceThread)newShow).enableTint(90, 100); // hue (0-360), brightness (0-100)
				}else{
					if(currentSensorShow == 0){
						newShow = new ImageSequenceThread(fixture, null, 60, detectorMngr.getFps(), raster, "ImageSequenceThread", ShowThread.LOW, imageCache.getSequence("redThrob"), false);					
						((ImageSequenceThread)newShow).enableTint(90, 100); // hue (0-360), brightness (0-100)
					} else if(currentSensorShow == 1){
						newShow = new ThrobbingThread(fixture, null, 60, detectorMngr.getFps(), raster, "ThrobbingThread", ShowThread.LOW, 255, 0, 0, 500, 500, 0, 0);			
					} else if(currentSensorShow == 2){
						newShow = new PropellerThread(fixture, null, 5, detectorMngr.getFps(), raster, "PropellerThread", ShowThread.LOW, 255, 0, 0, 5, 10);
					} else if(currentSensorShow == 3){
						newShow = new SpiralThread(fixture, null, 20, detectorMngr.getFps(), raster, "SpiralThread", ShowThread.LOW, 0, 255, 255, 10, 10, 0.5f, 100, guiWindow.gui.loadImage("depends//images//sprites//sphere50alpha.png"));
					} else if(currentSensorShow == 4){
						float[][] colorlist = new float[3][3];
						colorlist[0][0] = 255;
						colorlist[0][1] = 0;
						colorlist[0][2] = 0;
						colorlist[1][0] = 255;
						colorlist[1][1] = 255;
						colorlist[1][2] = 0;
						colorlist[2][0] = 255;
						colorlist[2][1] = 0;
						colorlist[2][2] = 0;
						float[] pointlist = new float[3];
						pointlist[0] = 0;
						pointlist[1] = 0.3f;
						pointlist[2] = 1;
						ColorScheme spectrum = new ColorScheme(colorlist, pointlist);
						newShow = new DartBoardThread(fixture, null, 20, detectorMngr.getFps(), raster, "DartBoardThread", ShowThread.LOW, spectrum);
					} else if(currentSensorShow == 5){
						newShow = new PieThread(fixture, null, 10, detectorMngr.getFps(), raster, "PieThread", ShowThread.LOW, 255, 255, 0, guiWindow.gui.loadImage("depends//images//sprites//bar40alpha.png"));
					}
				}

				// everything happens in here now.
				startShow(newShow);				
			}				
		}
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
		System.out.println("got stop from:\t" + showthread);
		liveShows.remove(showthread);
		availableFixtures.addAll(returnedFlowers);
		System.out.println("currently there are still " + liveShows.size() + " running and " + availableFixtures.size() + " fixtures unallocated");
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

			System.out.println("starting:\t" + newshow);
			//System.out.println("priority:\t" + newshow.getShowPriority());
			
			((GUI) guiWindow.gui).addActiveShow(newshow);
			newshow.start();
		}
	}
	
	//**************************************************************************
	public void tempUpdate(float update){
		/**
		 * TODO: Used for checking temperature of installation server.
		 */
	}
	
	public void timedEvent(TimedEvent e){
		//System.out.println(e.hour+":"+e.minute+":"+e.sec);
		PGraphics2D raster = new PGraphics2D(256,256,null);
		ShowThread newShow = new Glockenspiel(fixtures, soundManager, 5, detectorMngr.getFps(), raster, "Glockenspiel", ShowThread.HIGHEST, e.hour, e.minute, e.sec);
		startShow(newShow);
	}
	
	public void weatherChanged(WeatherChangedEvent wce){
		if(wce.hasSunriseChanged()) {
			Calendar sunrise = wce.getRecord().getSunrise();
			int h = sunrise.get(Calendar.HOUR_OF_DAY);
			int m = sunrise.get(Calendar.MINUTE);
			int s = sunrise.get(Calendar.SECOND);
			System.out.println("Sunrise at " + h + ":" + m + ":" + s);
			//sunriseOn.reschedule(h-1, m, s); // turn off an hour before sunrise
		}
		if(wce.hasSunsetChanged()) {
			Calendar sunset = wce.getRecord().getSunset();
			int h = sunset.get(Calendar.HOUR_OF_DAY);
			int m = sunset.get(Calendar.MINUTE);
			int s = sunset.get(Calendar.SECOND);
			System.out.println("Sunset at " + h + ":" + m + ":" + s);
			//sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
		}

		System.out.println("CONDITION = " + wce.getRecord().getCondition());
		System.out.println("VISIBILITY = " + wce.getRecord().getVisibility());
		System.out.println("OUTSIDE TEMP = " + wce.getRecord().getOutsideTemperature());
	}
	
	public static void main(String[] args) {					// PROGRAM LAUNCH
		new Conductor(args);
	}


}