package net.electroland.laface.core;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.sql.Timestamp;
import java.util.Calendar;
import java.util.Collection;
import java.util.Properties;
import javax.swing.JFrame;

import processing.core.PConstants;
import processing.core.PImage;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.laface.scheduler.TimedEvent;
import net.electroland.laface.scheduler.TimedEventListener;
import net.electroland.laface.gui.ControlPanel;
import net.electroland.laface.gui.RasterPanel;
import net.electroland.laface.shows.Blackout;
import net.electroland.laface.shows.DrawTest;
import net.electroland.laface.shows.Floaters;
import net.electroland.laface.shows.Highlighter;
import net.electroland.laface.shows.ImageSequence;
import net.electroland.laface.shows.Reflection2;
import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.laface.tracking.Tracker;
import net.electroland.laface.weather.WeatherChangeListener;
import net.electroland.laface.weather.WeatherChangedEvent;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.laface.weather.WeatherChecker;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class LAFACEMain extends JFrame implements AnimationListener, ActionListener, TimedEventListener, WeatherChangeListener {
	
	public DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	public AnimationManager amr;
	private Properties lightProps;
	public RasterPanel rasterPanel;
	private ControlPanel controlPanel;
	private int guiWidth = 1056;	// TODO get from properties
	private int guiHeight = 310;
	private int lowCondition = 29;
	private float lowVisibility = 8.0f;
	public Raster firstRaster, secondRaster, thirdRaster;
	public CarTracker carTracker;
	public PImage highlight, linearGradient, leftarrow, rightarrow, verticalGradient;
	public Tracker tracker;
	public ImageSequenceCache imageCache;	// only needed for testing
	private WeatherChecker weatherChecker;
	private TimedEvent sunriseOn = new TimedEvent(5,00,00, this); // on at sunrise-1 based on weather
	private TimedEvent middayOff = new TimedEvent(10,00,00, this); // off at 10 AM for sun reasons
	private TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather
	private TimedEvent nightOff = new TimedEvent(1,00,00, this); // off at 1 AM
	private TimedEvent clockEvents[];		// used for periodic status output to console
	
	public LAFACEMain() throws UnknownHostException, OptionException{
		super("LAFACE Control Panel");
		setLayout(new MigLayout("insets 0 0 0 0"));
		setSize(guiWidth, guiHeight);
		
		lightProps = loadProperties("depends//lights.properties");
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				System.exit(0);								// closes app
			}
		});
		
		rasterPanel = new RasterPanel(this, dmr.getRecipients(), 174, 7);
		//Raster raster = getRaster();
		
		highlight = rasterPanel.loadImage("depends//images//highlight.png");
		linearGradient = rasterPanel.loadImage("depends//images//linear.png");
		leftarrow = rasterPanel.loadImage("depends//images//leftarrow.png");
		rightarrow = rasterPanel.loadImage("depends//images//rightarrow.png");
		verticalGradient = rasterPanel.loadImage("depends//images//linear_vertical.png");

		firstRaster = getRaster();	// first wave show
		secondRaster = getRaster();	// second wave show
		thirdRaster = getRaster();	// transition show
		rasterPanel.setRaster(firstRaster);
		rasterPanel.setMinimumSize(new Dimension(1048,133));
		add(rasterPanel, "wrap");
		controlPanel = new ControlPanel(this);
		add(controlPanel, "wrap");
		
		clockEvents = new TimedEvent[24*60];	// event every minute just for testing
		for(int h=0; h<24; h++){
			for(int m=0; m<60; m++){
				clockEvents[(h+1)*m] = new TimedEvent(h,m,0,this);
			}
		}

		// this gets rid of exception for not using native acceleration
		System.setProperty("com.sun.media.jai.disableMediaLib", "true");
		
		// this is running the blob tracker server
		//tracker = new Tracker(this, 3);
		//tracker.start();
		
		//Animation a = new WaveShow(firstRaster);
		//Wave wave = new Wave(0, firstRaster, 0, 0);
		//((WaveShow)a).addWave(0, wave);
		//Wave newwave = new Wave(1, firstRaster, 0, 0);
		//newwave.setAlpha(0);	// start second wave invisible
		//((WaveShow)a).addWave(1, newwave);
		// ADD ANY ADDITIONAL WAVES HERE
		
		Collection<Recipient> fixtures = dmr.getRecipients();
		//amr.startAnimation(a, fixtures); 					// start the wave show
		//Animation highlighter = new Highlighter(secondRaster, highlight);
		//Animation newa = new Reflection2(this, firstRaster, leftarrow, rightarrow);
		//Animation newa = new Floaters(this, firstRaster, verticalGradient);
		
		try {
			Properties imageProps = new Properties();
			imageProps.load(new FileInputStream(new File("depends//images.properties")));
			imageCache = new ImageSequenceCache(imageProps, rasterPanel);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		// TODO uncomment this to test direct tracking video
		Animation newa = new ImageSequence(firstRaster, imageCache.getSequence("test"), true);
		
		amr.startAnimation(newa, fixtures);
		amr.goLive(); 
		controlPanel.refreshWaveList();
		//tracker.addTrackListener((TrackListener) highlighter);	// highlighter displays locations
		
		setResizable(true);
		setVisible(true);
		rasterPanel.init();
		
		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
	}
	
	public Properties loadProperties(String filename){
		try{
			lightProps = new Properties();
			lightProps.load(new FileInputStream(new File(filename)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lightProps;
	}
	
	public Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.faceRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(rasterPanel.createGraphics(width, height, PConstants.P3D));
	}
	
	public Animation getCurrentAnimation(){
		return amr.getCurrentAnimation(dmr.getRecipient("face0"));
	}
	
	public int getCurrentWaveID(){
		return controlPanel.getCurrentWaveID();
	}
	
	public void actionPerformed(ActionEvent e) {	// Respond to JFrame event
		Animation a = amr.getCurrentAnimation(dmr.getRecipient("face0"));
		if(a instanceof WaveShow){
			//String[] event = e.getActionCommand().split(":");
			
		} else if(a instanceof DrawTest){
			String[] event = e.getActionCommand().split(":");
			if(event[0].equals("turnOn")){
				((DrawTest)a).turnOn(Integer.parseInt(event[1]));
			} else if(event[0].equals("turnOff")){
				((DrawTest)a).turnOff(Integer.parseInt(event[1]));
			}
		}
	}

	public void completed(Animation a) {	// Respond to animation ending
		System.out.println("animation " + a + " completed!");
	}

	public void timedEvent(TimedEvent event) {
		if(event == sunriseOn) {			// activate
			amr.goLive();
		} else if (event == middayOff) {	// deactivate
			amr.stop();
			dmr.blackOutAll();
		} else if (event == sunsetOn){		// activate
			amr.goLive();
		} else if (event == nightOff){		// deactivate
			amr.stop();
			dmr.blackOutAll();
		} else {
			System.out.println(new Timestamp(System.currentTimeMillis()).toString());
		}
	}
	
	public void tempUpdate(float tu) {
		// check temperature of the system enclosure
	}

	public void weatherChanged(WeatherChangedEvent wce) {
		if(wce.hasSunriseChanged()) {
			Calendar sunrise = wce.getRecord().getSunrise();
			int h = sunrise.get(Calendar.HOUR_OF_DAY);
			int m = sunrise.get(Calendar.MINUTE);
			int s = sunrise.get(Calendar.SECOND);
			System.out.println("Sunrise at " + h + ":" + m + ":" + s);
			sunriseOn.reschedule(h-1, m, s); // turn on an hour before sunrise
			middayOff.reschedule(h+2, m, s); // turn off an hour after sunrise
		}
		if(wce.hasSunsetChanged()) {
			Calendar sunset = wce.getRecord().getSunset();
			int h = sunset.get(Calendar.HOUR_OF_DAY);
			int m = sunset.get(Calendar.MINUTE);
			int s = sunset.get(Calendar.SECOND);
			System.out.println("Sunset at " + h + ":" + m + ":" + s);
			sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
		}
		
		// if conditions are lower than 29 (mostly cloudy or worse) and vis is less than 10 miles, startup
		if (wce.getRecord().getCondition() < lowCondition && wce.getRecord().getVisibility() < lowVisibility) {
			// check if it's during the mid-day off gap
			if(Calendar.HOUR_OF_DAY >= middayOff.hour && Calendar.MINUTE >= middayOff.minute && Calendar.SECOND >= middayOff.sec){
				if(Calendar.HOUR_OF_DAY <= sunsetOn.hour && Calendar.MINUTE <= sunsetOn.minute && Calendar.SECOND <= sunsetOn.sec){
					amr.goLive();	// assumes a show had been started on application start
				}
			}
		}
	}
	
	
	
	
	public static void main(String[] args){
		try {
			new LAFACEMain();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}
	
}


