package net.electroland.lafm.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Calendar;
import java.util.Iterator;
import java.util.Properties;
import java.util.Vector;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;
import net.electroland.lafm.gui.GUI;
import net.electroland.lafm.gui.GUIWindow;
import net.electroland.lafm.scheduler.TimedEvent;
import net.electroland.lafm.scheduler.TimedEventListener;
import net.electroland.lafm.shows.Glockenspiel;
import net.electroland.lafm.shows.ImageSequenceThread;
import net.electroland.lafm.shows.ThrobbingThread;
import net.electroland.lafm.weather.WeatherChangeListener;
import net.electroland.lafm.weather.WeatherChangedEvent;
import net.electroland.lafm.weather.WeatherChecker;
import processing.core.PGraphics2D;
import promidi.MidiIO;
import promidi.Note;

public class Conductor extends Thread implements ShowThreadListener, WeatherChangeListener, TimedEventListener{
	
	public GUIWindow guiWindow;				// frame for GUI
	static public DMXLightingFixture[] flowers;		// all flower fixtures
	public MidiIO midiIO;						// sensor data IO
	public DetectorManager detectorMngr;
	public SoundManager soundManager;
	public WeatherChecker weatherChecker;
	public Properties sensors;					// pitch to fixture mappings
	public TimedEvent[] clockEvents;
	private ImageSequenceCache imageCache; // for ImageSequenceThreads

	// sample timed events, but I assume the building will be closed for some time at night
	//TimedEvent sunriseOn = new TimedEvent(6,00,00, this); // on at sunrise-1 based on weather
	//TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather

	static private Vector <ShowThread>liveShows;
	private Vector <DMXLightingFixture> usedFixtures;

	public Conductor(String args[]){
	
		// to track which fixtures are used, and what shows are currently running.
		liveShows = new Vector<ShowThread>();
		usedFixtures = new Vector<DMXLightingFixture>();
		
		// maybe move this to a static method.
		Properties lightProps = new Properties();
		try {

			// load props
			lightProps.load(new FileInputStream(new File("depends//lights.properties")));

			// set up fixtures
			detectorMngr = new DetectorManager(lightProps);

			// get fixtures
			flowers = detectorMngr.getFixtures();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
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
		guiWindow = new GUIWindow(this, lightProps);
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
	}
	
	static public void makeShow(ShowThread show){
		// this is temporary for doing diagnostics via the GUI
		liveShows.add(show);
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
				PGraphics2D raster = new PGraphics2D(256,256,null);
				//ShowThread newShow = new DiagnosticThread(fixture, null, 60, detectorMngr.getFps(), raster);
				ShowThread newShow;
				if (note.getPitch() == 36){
					System.out.println("new 'redThrob' ImageSequenceThread");
					newShow = new ImageSequenceThread(fixture, null, 60, detectorMngr.getFps(), raster, "ImageSequenceThread", imageCache.getSequence("redThrob"), false);					
				}else{
					System.out.println("new ThrobbingThread");
					newShow = new ThrobbingThread(fixture, null, 60, detectorMngr.getFps(), raster, "ThrobbingThread", 255, 0, 0, 500, 500, 0, 0);					
				}

				// manage threadpool
				liveShows.add(newShow);
				usedFixtures.add(fixture);
				((GUI) guiWindow.gui).addActiveShow(newShow);

				// tell thread that we want to be notified of it's end.
				newShow.addListener(this);

				newShow.start();

			}else{
				// off events
				// do nothing for now.
			}
		}
	}	
	
	// Any time a show is done, this call back will be called, so that
	// conductor knows the show is over, and that the flowers are available
	// for reallocation.
	public void notifyComplete(ShowThread showthread, DMXLightingFixture[] returnedFlowers) {
		liveShows.remove(showthread);
		((GUI) guiWindow.gui).removeActiveShow(showthread.getID());
		for (int i = 0; i < returnedFlowers.length; i++){
			usedFixtures.remove(returnedFlowers[i]);
		}
	}
	
	public void tempUpdate(float update){
		/**
		 * TODO: Used for checking temperature of installation server.
		 */
	}
	
	public void timedEvent(TimedEvent e){
		System.out.println(e.hour+":"+e.minute+":"+e.sec);
		DMXLightingFixture[] fixtures = detectorMngr.getFixtures();
		PGraphics2D raster = new PGraphics2D(256,256,null);
		ShowThread newShow = new Glockenspiel(fixtures, soundManager, 5, detectorMngr.getFps(), raster, "Glockenspiel", e.hour, e.minute, e.sec);
		//ShowThread newShow = new ThrobbingThread(fixtures, null, 5, detectorMngr.getFps(), raster, "ThrobbingThread", 255, 0, 0, 500, 500, 0, 0);
		liveShows.add(newShow);
		//usedFixtures.addAll();	// need a Collection of fixtures
		((GUI) guiWindow.gui).addActiveShow(newShow);
		newShow.addListener(this);
		newShow.start();
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