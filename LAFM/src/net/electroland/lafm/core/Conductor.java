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
import net.electroland.lafm.gui.GUIWindow;
import net.electroland.lafm.scheduler.TimedEvent;
import net.electroland.lafm.scheduler.TimedEventListener;
import net.electroland.lafm.shows.DiagnosticThread;
import net.electroland.lafm.weather.WeatherChangeListener;
import net.electroland.lafm.weather.WeatherChangedEvent;
import net.electroland.lafm.weather.WeatherChecker;
import processing.core.PGraphics2D;
import processing.core.PImage;
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
		String filename = (args.length > 0) ? args[0] : "depends//lights.properties";
		Properties p = new Properties();
		try {

			// load props
			p.load(new FileInputStream(new File(filename)));

			// set up fixtures
			detectorMngr = new DetectorManager(p);

			// get fixtures
			flowers = detectorMngr.getFixtures();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		filename = "depends//sensors.properties";
		sensors = new Properties();
		try{

			// load sensor info
			sensors.load(new FileInputStream(new File(filename)));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		midiIO = MidiIO.getInstance();
		try{
			midiIO.plug(this, "midiEvent", 0, 0);	// device # and midi channel
		} catch(Exception e){
			e.printStackTrace();
		}
		soundManager = new SoundManager();
		guiWindow = new GUIWindow(this);
		guiWindow.setVisible(true);
		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
		//weatherChecker.start();
	}
	
	static public void makeShow(ShowThread show){
		// this is temporary for doing diagnostics via the GUI
		liveShows.add(show);
	}
		
	public void midiEventWorking(Note note){
		// flower sensor is activated
		System.out.println("MIDI event: "+note.getPitch()+" "+note.getVelocity());

		try{

			int fixturenum = Integer.valueOf(sensors.getProperty(String.valueOf(note.getPitch())));
			boolean on = note.getVelocity() == 0 ? false : true;

			/*
			if (!usedFixtures.contains(flowers[fixturenum])){
				ShowThread newShow = new DiagnosticThread(flowers[fixturenum],
						null, Integer.MAX_VALUE, 30, new PGraphics2D(256, 256, null));
				liveShows.add(newShow);
				usedFixtures.add(flowers[fixturenum]);
			}
			*/
			
			PGraphics2D raster = new PGraphics2D(256,256,null);
			if(on){
				raster.background(-1);				
			} else {
				raster.background(-16777216);
			}
			String[] fixtures = this.detectorMngr.getFixtureIds();
			for(int i=0; i<fixtures.length; i++){
				if(fixtures[i].equals("fixture"+fixturenum)){
					System.out.println(on+" fixture"+fixturenum);
					this.detectorMngr.getFixture("fixture"+fixturenum).sync((PImage)raster);
					break;
				}
			}
			
			// tell each show that an event has happened
			Iterator<ShowThread> i = liveShows.iterator();
			while (i.hasNext()){
				ShowThread s = i.next();
				if (s instanceof SensorListener){
					((SensorListener)s).sensorEvent(flowers[fixturenum], on);
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
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
				ShowThread newShow = new DiagnosticThread(fixture,
						null, 60, detectorMngr.getFps(), raster);

				// manage threadpool
				liveShows.add(newShow);
				usedFixtures.add(fixture);		

				// tell thread that we won't to be notified of it's end.
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
		/**
		 * TODO: Reacts to all timed events (15, 30, 60 minute schedule + sunset/sunrise).
		 */
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