package net.electroland.enteractive.core;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.sql.Timestamp;
import java.util.Calendar;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.apache.log4j.Logger;

import net.electroland.enteractive.gui.GUI;
import net.electroland.enteractive.gui.Lights3D;
import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;
import net.electroland.enteractive.shows.Blackout;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.enteractive.shows.MusicBox;
import net.electroland.enteractive.shows.Plasma;
import net.electroland.enteractive.shows.Pong;
import net.electroland.enteractive.shows.Spotlight;
import net.electroland.enteractive.weather.WeatherChangeListener;
import net.electroland.enteractive.weather.WeatherChangedEvent;
import net.electroland.enteractive.weather.WeatherChecker;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.lighting.detector.animation.transitions.LinearFade;
import net.electroland.udpUtils.TCUtil;
import net.electroland.udpUtils.UDPParser;
import net.electroland.util.OptionException;
import net.miginfocom.swing.MigLayout;
import processing.core.PConstants;
import processing.core.PImage;

@SuppressWarnings("serial")
public class EnteractiveMain extends JFrame implements AnimationListener, ActionListener, TimedEventListener, ModelListener, WeatherChangeListener{
	
	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private AnimationManager amr;
	private SoundManager smr;
	private Lights3D lights3D;
	private GUI gui;
	private TCUtil tcu;
	private PersonTracker ptr;
	private UDPParser udp;
	private Properties lightProps, systemProps;
	private WeatherChecker weatherChecker;
	private int guiWidth = 180;	// TODO get from properties
	private int guiHeight = 110;
	private int lowCondition = 29;
	private float lowVisibility = 8.0f;
	private TimedEvent sunriseOn = new TimedEvent(5,00,00, this); 	// on at sunrise-1 based on weather
	private TimedEvent middayOff = new TimedEvent(11,00,00, this); 	// off at 11 AM for sun reasons
	private TimedEvent sunsetOn = new TimedEvent(16,00,00, this); 	// on at sunset-1 based on weather
	private TimedEvent nightOff = new TimedEvent(1,00,00, this); 		// off at 1 AM
	private Timestamp sunrise,midday,sunset,night;	// these get updated whenever the weather checker updates timed events
	private String[] animationList;
	private JComboBox animationDropDown, displayDropDown, rasterDropDown, sensorDropDown;
	private JButton printSensorActivityButton, grabWebcamImage, testSound;
	PImage rippleTexture, sweepTexture, sphereTexture, propellerTexture, spiralTexture, radarTexture;
	PImage ballTexture, pongTitle;	// pong textures
	
	static Logger logger = Logger.getLogger(EnteractiveMain.class);
	
	public EnteractiveMain(String[] args) throws UnknownHostException, OptionException{
		super("Enteractive Control Panel");
		
		// need this call here to access the local log4j properties first, before the props in any linked project
		logger.info("Enteractive Startup");

		systemProps = loadProperties("depends//enteractive.properties");
		lightProps = loadProperties("depends//lights.properties");
		int fps = Integer.parseInt(lightProps.getProperty("fps"));
		
		animationList = new String[2];
		animationList[0] = "LilyPad";
		animationList[1] = "Spotlight";
		
		dmr = new DetectorManager(lightProps); 				// requires loading properties
		dmp = new DetectorManagerJPanel(dmr);				// panel that renders the filters
		amr = new AnimationManager(dmp, fps);				// animation manager
		amr.addListener(this);								// let me know when animations are complete
		smr = new SoundManager();
		//smr = null;
		//add(dmp,"wrap");
		
		tcu = new TCUtil();									// tile controller utilities
		ptr = new PersonTracker(16,11);						// person tracker manages the model (x/y tile dimensions)
		ptr.getModel().addListener(this);					// add this as a listener
		
		try {
			udp = new UDPParser(10011, tcu, ptr);			// open our UDP receiver and begin parsing
			udp.start();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}

		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {		// when the X is hit in the frame
				smr.shutdown();
				System.exit(0);								// closes app
			}
		});
		/*
		Runtime.getRuntime().addShutdownHook(new Thread() {
		    public void run() { tcu.billyJeanMode(); }
		});
		*/
		

		lights3D = new Lights3D(600,600, dmr.getRecipient("floor"),  dmr.getRecipient("face"), ptr.getModel(), tcu);
		lights3D.setMinimumSize(new Dimension(600,600));
		gui = new GUI(guiWidth,guiHeight, dmr.getRecipient("floor"),  dmr.getRecipient("face"));
		Raster raster = getRaster();
		((GUI)gui).setRaster(raster);

		drawLayout();
		lights3D.init();
		gui.init();
		
		
		// LOAD IMAGE SPRITES FOR SHOWS
		rippleTexture = gui.loadImage("depends//images//ripple.png");
		sweepTexture = gui.loadImage("depends//images//sweep.png");
		sphereTexture = gui.loadImage("depends//images//sphere.png");
		propellerTexture = gui.loadImage("depends//images//propeller.png");
		spiralTexture = gui.loadImage("depends//images//spiral.png");
		ballTexture = gui.loadImage("depends//images//ball.png");
		pongTitle = gui.loadImage("depends//images//pongtitle.png");
		radarTexture = gui.loadImage("depends//images//radar.png");
		
		//currentAnimation = new ExampleAnimation(ptr.getModel(), raster, smr);
		//Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
		Animation a = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture, radarTexture);
		//Animation a = new MusicBox(ptr.getModel(), raster, smr);
		//Animation a = new Pong(ptr.getModel(), raster, smr, sphereTexture, pongTitle);
		//Animation a = new Plasma(raster);
		Collection<Recipient> fixtures = dmr.getRecipients();
		
		Timestamp now = new Timestamp(System.currentTimeMillis());
		sunrise = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunriseOn.hour, sunriseOn.minute, sunriseOn.sec, 0);
		midday = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), middayOff.hour, middayOff.minute, middayOff.sec, 0);
		sunset = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunsetOn.hour, sunsetOn.minute, sunsetOn.sec, 0);
		night = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), nightOff.hour, nightOff.minute, nightOff.sec, 0);
		//System.out.println(now.toString() +" "+ midday.toString() +" "+ sunset.toString());
		
		if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
			// if it is during an off period, remove the face recipient from fixtures
			fixtures.remove(dmr.getRecipient("face"));
			//System.out.println("DON'T TURN ON THE FACE!");
		}
		
		amr.startAnimation(a, fixtures); 					// start a show now, on this list of fixtures.
		amr.goLive(); 										// the whole system does nothing unless you "start" it.

		// wait 6 secs (for things to get started up) then check weather once a day
		weatherChecker = new WeatherChecker(6000, 24 * 60 * 60 * 1000);
		weatherChecker.addListener(this);
		weatherChecker.start();
	}

	public void completed(Animation a) {
		Timestamp now = new Timestamp(System.currentTimeMillis());
		
		// switch to a new animation
		System.out.println("animation " + a + " completed!");
		if (a instanceof Spotlight || a instanceof Pong){
			Raster raster = getRaster();
			((GUI)gui).setRaster(raster);
			Animation next = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture, radarTexture);
			Collection<Recipient> fixtures = dmr.getRecipients();
			if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
				// if it is during an off period, remove the face recipient from fixtures
				fixtures.remove(dmr.getRecipient("face"));
			}
			amr.startAnimation(next, fixtures);	
		}
	}

	public void actionPerformed(ActionEvent e) {
		//System.out.println(e.getActionCommand());
		
		if(e.getActionCommand().equals("comboBoxChanged")){
			if((JComboBox)e.getSource() == animationDropDown){
				if(animationDropDown.getSelectedItem() == "LilyPad"){
					Recipient floor = dmr.getRecipient("floor");
					if(!(amr.getCurrentAnimation(floor) instanceof LilyPad)){			// if not already lilypad
						Raster raster = getRaster();
						((GUI)gui).setRaster(raster);
						Animation a = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture, radarTexture);
						Collection<Recipient> fixtures = dmr.getRecipients();
						Animation transition = new LinearFade(2, getRaster());
						amr.startAnimation(a, transition, fixtures); 					// START LILYPAD
					}
			    } else if(animationDropDown.getSelectedItem() == "Spotlight"){			// if not already spotlight
			    	Recipient floor = dmr.getRecipient("floor");
					if(!(amr.getCurrentAnimation(floor) instanceof Spotlight)){
						Raster raster = getRaster();
						((GUI)gui).setRaster(raster);
						Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
						Collection<Recipient> fixtures = dmr.getRecipients();
						Animation transition = new LinearFade(2, getRaster());
						amr.startAnimation(a, transition, fixtures); 					// START SPOTLIGHT (30 secs)
					}
			    }
			} else if((JComboBox)e.getSource() == displayDropDown){
				if(displayDropDown.getSelectedItem() == "Comparison"){
					lights3D.setVisible(true);
					lights3D.setMode(1);
			    } else if(displayDropDown.getSelectedItem() == "Real World"){
			    	lights3D.setVisible(true);
					lights3D.setMode(2);
			    } else if(displayDropDown.getSelectedItem() == "Disabled"){
			    	lights3D.setVisible(false);
			    }
			} else if((JComboBox)e.getSource() == rasterDropDown){
				if(rasterDropDown.getSelectedItem() == "Face Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(1);
				} else if(rasterDropDown.getSelectedItem() == "Floor Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(2);
				} else if(rasterDropDown.getSelectedItem() == "No Detectors"){
					gui.setVisible(true);
					gui.setDetectorMode(0);
				} else if(rasterDropDown.getSelectedItem() == "Disabled"){
					gui.setVisible(false);
				}
			} else if((JComboBox)e.getSource() == sensorDropDown){
				if(sensorDropDown.getSelectedItem() == "Current Activity"){
					lights3D.setSensorMode(0);
				} else if(sensorDropDown.getSelectedItem() == "Averages"){
					lights3D.setSensorMode(1);
				} else if(sensorDropDown.getSelectedItem() == "Total Activity"){
					lights3D.setSensorMode(2);
				}
			}
		
		} else if((JButton)e.getSource() == printSensorActivityButton){
			int maxActivity = 0;
			int minActivity = 100000000;	// TODO kludge
			int sumActivity = 0;
			int tileCount = 0;
			// loop through all the tiles to get the max value
			List<TileController> tileControllers = tcu.getTileControllers();
			Iterator<TileController> iter = tileControllers.iterator();
			while(iter.hasNext()){
				TileController tc = iter.next();
				List<Tile> tiles = tc.getTiles();
				Iterator<Tile> tileiter = tiles.iterator();
				while(tileiter.hasNext()){
					Tile tile = tileiter.next();
					if(tile.getActivityCount() > maxActivity){
						maxActivity = tile.getActivityCount();
					} else if(tile.getActivityCount() < minActivity){
						minActivity = tile.getActivityCount();
					}
					sumActivity += tile.getActivityCount();
					tileCount++;
				}
			}
			System.out.println(new Timestamp(System.currentTimeMillis()).toString() +" Min Activity: "+ minActivity +", Max Activity: "+ maxActivity + ", Avg Activity: "+((float)sumActivity)/tileCount);
		} else if ((JButton)e.getSource() == grabWebcamImage){
			tcu.grabWebcamImage();
		} else if ((JButton)e.getSource() == testSound){ //2014
			//2014 testing
			smr.createMonoSound(smr.soundProps.getProperty("shooter"), 0, 0, 10, 10);
		}
		//Animation next = new AnotherAnimation(m, getRaster(), smr); 			// some fake animation
		//amr.startAnimation(next, new FadeTransition(5), dmr.getFixtures()); 	// some fake transition with a 5 second fade
	}
	
	private void drawLayout(){
		MigLayout layout = new MigLayout("insets 0 0 0 0, gap 1!");
		setLayout(layout);
		setBackground(Color.black);
		setForeground(Color.white);
		add(lights3D, "cell 0 0 1 3");
		
		JPanel controlPanel = new JPanel(new MigLayout());
		controlPanel.setBackground(Color.black);
		controlPanel.setForeground(Color.white);
		controlPanel.add(new JLabel("Current Animation:"), "wrap");
		
		// drop down list to select current animation
		animationDropDown = new JComboBox(animationList);
		//animationDropDown.setBackground(Color.black);
		//animationDropDown.setForeground(Color.white);
		animationDropDown.setMinimumSize(new Dimension(180, 20));
		animationDropDown.addActionListener(this);		
		controlPanel.add(animationDropDown, "wrap");
		
		controlPanel.add(new JLabel("3D Mode:"), "wrap");
		
		// drop down list to select 3d mode
		displayDropDown = new JComboBox(new String[] {"Comparison", "Real World", "Disabled"});
		//displayDropDown.setBackground(Color.black);
		//displayDropDown.setForeground(Color.white);
		displayDropDown.setMinimumSize(new Dimension(180, 20));
		displayDropDown.addActionListener(this);		
		controlPanel.add(displayDropDown, "wrap");
		
		controlPanel.add(new JLabel("Raster Mode:"), "wrap");
		
		// drop down list to select raster mode
		rasterDropDown = new JComboBox(new String[] {"Face Detectors", "Floor Detectors", "No Detectors", "Disabled"});
		//rasterDropDown.setBackground(Color.black);
		//rasterDropDown.setForeground(Color.white);
		rasterDropDown.setMinimumSize(new Dimension(180, 20));
		rasterDropDown.addActionListener(this);		
		controlPanel.add(rasterDropDown, "wrap");
		
		controlPanel.add(new JLabel("Sensor Mode:"), "wrap");
		
		// drop down list to select sensor display mode
		sensorDropDown = new JComboBox(new String[] {"Current Activity", "Averages", "Total Activity"});
		//sensorDropDown.setBackground(Color.black);
		//sensorDropDown.setForeground(Color.white);
		sensorDropDown.setMinimumSize(new Dimension(180, 20));
		sensorDropDown.addActionListener(this);		
		controlPanel.add(sensorDropDown, "wrap");
		
		printSensorActivityButton = new JButton("Print Sensor Activity");
		printSensorActivityButton.addActionListener(this);
		printSensorActivityButton.setMinimumSize(new Dimension(180, 20));
		//printSensorActivityButton.setMaximumSize(new Dimension(180, 20));
		controlPanel.add(printSensorActivityButton, "wrap");
		
		grabWebcamImage = new JButton("Grab Webcam Image");
		grabWebcamImage.addActionListener(this);
		grabWebcamImage.setMinimumSize(new Dimension(180, 20));
		controlPanel.add(grabWebcamImage, "wrap");

		testSound = new JButton("Play Test Sound");
		testSound.addActionListener(this);
		testSound.setMinimumSize(new Dimension(180, 20));
		controlPanel.add(testSound, "wrap");
		
		
		add(controlPanel, "cell 1 0, width 200!, height 380!, gap 0!");
		
		JPanel rasterPanel = new JPanel();
		rasterPanel.setBackground(Color.black);
		rasterPanel.setForeground(Color.white);
		rasterPanel.add(gui);
		add(rasterPanel, "cell 1 1, width 200!, height 120!, gap 0!");
		
		JPanel placeHolder3 = new JPanel();
		placeHolder3.setBackground(Color.black);
		placeHolder3.setForeground(Color.white);
		//placeHolder3.add(new JLabel("Audio Levels Go Here"));
		add(placeHolder3, "cell 1 2, width 200!, height 90!, gap 0!");
		
		setSize(800, 620);
		if(Boolean.parseBoolean(systemProps.getProperty("headless"))){
			setVisible(false);
		} else {
			setVisible(true);
		}
		setResizable(true);
	}
	
	private Raster getRaster(){
		String[] dimensions = lightProps.getProperty("raster.tileRaster").split(" ");
		float multiplier = Float.parseFloat(lightProps.getProperty("rasterDimensionScaling"));
		int width = (int)(Integer.parseInt(dimensions[1]) * multiplier);
		int height = (int)(Integer.parseInt(dimensions[3]) * multiplier);
		return new Raster(gui.createGraphics(width, height, PConstants.P3D));
	}
	
	public Properties loadProperties(String file) {
		Properties newprops = new Properties();
		try{
			newprops = new Properties();
			newprops.load(new FileInputStream(new File(file)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return newprops;
	}

	public void timedEvent(TimedEvent event) {
		// TODO activate/deactivate the face of the building
		System.out.println("timed event");
		if(event == sunriseOn) {			// activate
			Recipient floor = dmr.getRecipient("floor");
			amr.startAnimation(amr.getCurrentAnimation(floor), dmr.getRecipients());
		} else if (event == middayOff) {	// deactivate
			Recipient face = dmr.getRecipient("face");
			amr.reapRecipient(face).blackOut();
		} else if (event == sunsetOn){		// activate
			Recipient floor = dmr.getRecipient("floor");
			amr.startAnimation(amr.getCurrentAnimation(floor), dmr.getRecipients());
		} else if (event == nightOff){		// deactivate
			Recipient face = dmr.getRecipient("face");
			amr.reapRecipient(face).blackOut();
		}
	}

	public void modelEvent(ModelEvent e) {
		// check current time so we know if we should activate the face or not
		Timestamp now = new Timestamp(System.currentTimeMillis());
		
		if(e.getType() == Model.ModelConstants.FOUR_CORNERS){
			System.out.println("Four Corners!");
		} else if(e.getType() == Model.ModelConstants.OPPOSITE_CORNERS){
			System.out.println("Corners 1 and 4!");
			// switch to MUSICBOX show
			/*
			Recipient floor = dmr.getRecipient("floor");
			if(amr.getCurrentAnimation(floor) instanceof LilyPad){
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				Animation a = new MusicBox(ptr.getModel(), raster, smr, sweepTexture);
				Collection<Recipient> fixtures = dmr.getRecipients();
				if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
					// if it is during an off period, remove the face recipient from fixtures
					fixtures.remove(dmr.getRecipient("face"));
				}
				//Animation transition = new LinearFade(2, getRaster());
				amr.startAnimation(a, fixtures); 					// START MUSICBOX
			}
			*/
		} else if(e.getType() == Model.ModelConstants.OPPOSITE_CORNERS2){
			System.out.println("Corners 2 and 3!");
			// switch to PONG game
			/*
			Recipient floor = dmr.getRecipient("floor");
			if(amr.getCurrentAnimation(floor) instanceof LilyPad){
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				Animation a = new Pong(ptr.getModel(), raster, smr, ballTexture, pongTitle);
				Collection<Recipient> fixtures = dmr.getRecipients();
				if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
					// if it is during an off period, remove the face recipient from fixtures
					fixtures.remove(dmr.getRecipient("face"));
				}
				//Animation transition = new LinearFade(2, getRaster());
				amr.startAnimation(a, fixtures); 					// START PONG (3 points)
			}
			*/
		} else if(e.getType() == Model.ModelConstants.EMPTY){
			System.out.println("area empty");
			// switch to SCREENSAVER
			
			Recipient floor = dmr.getRecipient("floor");
			if(amr.getCurrentAnimation(floor) instanceof LilyPad || amr.getCurrentAnimation(floor) instanceof MusicBox || amr.getCurrentAnimation(floor) instanceof Spotlight){
				if(amr.getCurrentAnimation(floor) instanceof MusicBox){
					smr.killAll();
				}
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				//Animation a = new Plasma(raster);
				Animation a = new Spotlight(ptr.getModel(), raster, smr, sphereTexture);
				Collection<Recipient> fixtures = dmr.getRecipients();
				if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
					// if it is during an off period, remove the face recipient from fixtures
					fixtures.remove(dmr.getRecipient("face"));
				}
				Animation transition = new LinearFade(3, getRaster());
				amr.startAnimation(a, transition, fixtures);		// START SCREENSAVER
			}
		} else if(e.getType() == Model.ModelConstants.NOT_EMPTY){
			System.out.println("area re-activated");
			// switch back to LILYPAD
			
			Recipient floor = dmr.getRecipient("floor");
			//if(amr.getCurrentAnimation(floor) instanceof Plasma){
			//if(amr.getCurrentAnimation(floor) instanceof Spotlight){
				Raster raster = getRaster();
				((GUI)gui).setRaster(raster);
				Animation next = new LilyPad(ptr.getModel(), raster, smr, rippleTexture, sweepTexture, propellerTexture, spiralTexture, sphereTexture, radarTexture);
				Collection<Recipient> fixtures = dmr.getRecipients();
				if((now.after(night) && now.before(sunrise)) || (now.after(midday) && now.before(sunset))){
					// if it is during an off period, remove the face recipient from fixtures
					fixtures.remove(dmr.getRecipient("face"));
				}
				//Animation transition = new LinearFade(1, getRaster());
				amr.startAnimation(next, fixtures);		// START LILYPAD
			//}
		}
	}
	
	
	public static void main(String args[]){
		try {
			new EnteractiveMain(args);
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}

	public void tempUpdate(float tu) {
		// TODO used for monitoring temp of system
	}

	public void weatherChanged(WeatherChangedEvent wce) {
		if(wce.hasSunriseChanged()) {
			Calendar sunrise = wce.getRecord().getSunrise();
			int h = sunrise.get(Calendar.HOUR_OF_DAY);
			int m = sunrise.get(Calendar.MINUTE);
			int s = sunrise.get(Calendar.SECOND);
			System.out.println("Sunrise at " + h + ":" + m + ":" + s);
			sunriseOn.reschedule(h-1, m, s); // turn on an hour before sunrise
		}
		if(wce.hasSunsetChanged()) {
			Calendar sunset = wce.getRecord().getSunset();
			int h = sunset.get(Calendar.HOUR_OF_DAY);
			int m = sunset.get(Calendar.MINUTE);
			int s = sunset.get(Calendar.SECOND);
			System.out.println("Sunset at " + h + ":" + m + ":" + s);
			sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
		}
		
		Timestamp now = new Timestamp(System.currentTimeMillis());
		sunrise = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunriseOn.hour, sunriseOn.minute, sunriseOn.sec, 0);
		midday = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), middayOff.hour, middayOff.minute, middayOff.sec, 0);
		sunset = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), sunsetOn.hour, sunsetOn.minute, sunsetOn.sec, 0);
		night = new Timestamp(now.getYear(), now.getMonth(), now.getDate(), nightOff.hour, nightOff.minute, nightOff.sec, 0);
		
		// if conditions are lower than 29 (mostly cloudy or worse) and vis is less than 10 miles, startup
		if (wce.getRecord().getCondition() < lowCondition && wce.getRecord().getVisibility() < lowVisibility) {
			// check if it's during the mid-day off gap
			//Recipient floor = dmr.getRecipient("floor");
			//amr.startAnimation(amr.getCurrentAnimation(floor), dmr.getRecipients());
			
		}
	}
}
