package net.electroland.laface.core;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
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
import net.electroland.elvisVideoProcessor.ElProps;
import net.electroland.elvisVideoProcessor.LAFaceVideoListener;
import net.electroland.elvisVideoProcessor.LAFaceVideoProcessor;
import net.electroland.laface.scheduler.TimedEvent;
import net.electroland.laface.scheduler.TimedEventListener;
import net.electroland.laface.gui.ControlPanel;
import net.electroland.laface.gui.RasterPanel;
import net.electroland.laface.shows.DrawTest;
import net.electroland.laface.shows.Video;
import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.installutils.weather.WeatherChangeListener;
import net.electroland.installutils.weather.WeatherChangedEvent;
import net.electroland.installutils.weather.WeatherChecker;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;

@SuppressWarnings("serial")
public class LAFACEMain extends JFrame implements AnimationListener, ActionListener, TimedEventListener, WeatherChangeListener, LAFaceVideoListener {
	
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
	public PImage highlight, linearGradient, leftarrow, rightarrow, verticalGradient;
	public LAFaceVideoProcessor lafvp;
	public ImageSequenceCache imageCache;	// only needed for testing
	private WeatherChecker weatherChecker;
	private TimedEvent sunriseOn = new TimedEvent(5,00,00, this); // on at sunrise-1 based on weather
	//2011_06_07 changed this from 10 AM to whatever for testing
	private TimedEvent middayOff = new TimedEvent(10,00,00, this); // off at 10 AM for sun reasons
	private TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather
	private TimedEvent nightOff = new TimedEvent(1,00,00, this); // off at 1 AM
	private TimedEvent morningAdaptation = new TimedEvent(6,30,00, this);
	private TimedEvent nightAdaptation = new TimedEvent(20,00,00, this);
	private Timestamp sunrise,midday,sunset,night;	// these get updated whenever the weather checker updates timed events
	
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
		
		// START GRABBING CAMERA FRAMES
		ElProps.init("depends/LAFace.props");
		lafvp = new LAFaceVideoProcessor(ElProps.THE_PROPS);
		lafvp.setBackgroundAdaptation(ElProps.THE_PROPS.setProperty("adaptation", .1));
		try {
			lafvp.setSourceStream(ElProps.THE_PROPS.getProperty("camera", "axis"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		lafvp.addListener(this);
		lafvp.start();
		

		// this gets rid of exception for not using native acceleration
		//System.setProperty("com.sun.media.jai.disableMediaLib", "true");
		
		
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
		//Animation newa = new ImageSequence(firstRaster, imageCache.getSequence("test"), true);
		Animation newa = new Video(firstRaster, lafvp);
		
		amr.startAnimation(newa, fixtures);

		Timestamp now = new Timestamp(System.currentTimeMillis());
		sunrise = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunriseOn.hour, sunriseOn.minute, sunriseOn.sec, 0);
		midday = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), middayOff.hour, middayOff.minute, middayOff.sec, 0);
		sunset = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunsetOn.hour, sunsetOn.minute, sunsetOn.sec, 0);
		night = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), nightOff.hour, nightOff.minute, nightOff.sec, 0);
		
		// TODO uncomment this to have the display turn off during the day
		/*
		// animation manager only started if it's between the reasonable periods
		if((now.after(sunset) && now.before(night)) || (now.after(sunrise) && now.before(midday))){
			amr.goLive(); 	// START ANIMATION MANAGER
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " ImageSequence Show Started");
		} else {
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " Waiting for appropriate display time");
		}
		*/
		amr.goLive();

		controlPanel.refreshWaveList();
		//tracker.addTrackListener((TrackListener) highlighter);	// highlighter displays locations
		
		setResizable(true);
		setVisible(true);
		rasterPanel.init();
		
		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
		weatherChecker.start();
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
	
	public Raster getImageRaster(){
		String[] dimensions = lightProps.getProperty("raster.faceRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB));
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
		// TODO uncomment this to have the display go blank during the day
		
		if(event == sunriseOn) {			// activate
			dmr.turnOn();
			lafvp.setBackgroundAdaptation(Double.parseDouble(ElProps.THE_PROPS.getProperty("sunriseAdaptation")));
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " App Illuminating");
		} else if (event == morningAdaptation){
			lafvp.setBackgroundAdaptation(Double.parseDouble(ElProps.THE_PROPS.getProperty("morningAdaptation")));
		} else if (event == middayOff) {	// deactivate
			dmr.turnOff();
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " App Disabling Lights");
		} else if (event == sunsetOn){		// activate
			dmr.turnOn();
			lafvp.setBackgroundAdaptation(Double.parseDouble(ElProps.THE_PROPS.getProperty("sunsetAdaptation")));
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " App Illuminating");
		} else if (event == nightAdaptation){
			lafvp.setBackgroundAdaptation(Double.parseDouble(ElProps.THE_PROPS.getProperty("nightAdaptation")));
		} else if (event == nightOff){		// deactivate
			dmr.turnOff();
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " App Disabling Lights");
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
			morningAdaptation.reschedule(h, m, s);
		}
		if(wce.hasSunsetChanged()) {
			Calendar sunset = wce.getRecord().getSunset();
			int h = sunset.get(Calendar.HOUR_OF_DAY);
			int m = sunset.get(Calendar.MINUTE);
			int s = sunset.get(Calendar.SECOND);
			System.out.println("Sunset at " + h + ":" + m + ":" + s);
			sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
			nightAdaptation.reschedule(h+1, m, s);
		}
		
		Timestamp now = new Timestamp(System.currentTimeMillis());
		sunrise = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunriseOn.hour, sunriseOn.minute, sunriseOn.sec, 0);
		midday = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), middayOff.hour, middayOff.minute, middayOff.sec, 0);
		sunset = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunsetOn.hour, sunsetOn.minute, sunsetOn.sec, 0);
		night = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), nightOff.hour, nightOff.minute, nightOff.sec, 0);
		
		/*
		// TODO uncomment this to have the display go blank during the day
		// black out and pause the animation manager if during a black out period
		if(now.after(midday) && now.before(sunset)){
			amr.stop();
			dmr.blackOutAll();
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " ImageSequence Show Stopped");
		} else if(now.after(night) && now.before(sunrise)){
			amr.stop();
			dmr.blackOutAll();
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() + " ImageSequence Show Stopped");
		}
		*/
		
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
	

	public void cameraError(Exception cameraException) {
		// TODO switch to wave show
		System.out.println("camera error received");
		Raster raster = getRaster();
		rasterPanel.setRaster(raster);
		Animation a = new WaveShow(raster);
		Wave newwave = new Wave(0, raster, 0, 0);	// for shared wave sprite on multiple shows
		((WaveShow)a).addWave(0, newwave);
		Collection<Recipient> fixtures = dmr.getRecipients();
		System.out.println("attempting to start waves...");
		amr.startAnimation(a, fixtures); 
		System.out.println("wave show started");
		ImpulseThread impulseThread = new ImpulseThread(this);
		impulseThread.start();
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


