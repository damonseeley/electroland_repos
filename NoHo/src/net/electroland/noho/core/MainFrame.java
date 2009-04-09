package net.electroland.noho.core;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Calendar;
import java.util.Vector;

import javax.swing.JFrame;

import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoSouthCam;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.noho.graphics.AnimationManager;
import net.electroland.noho.graphics.Compositor;
import net.electroland.noho.graphics.ImageConsumer;
import net.electroland.noho.graphics.generators.sprites.LinearMotionRecSprite;
import net.electroland.noho.graphics.generators.sprites.SpriteImageGenerator;
import net.electroland.noho.graphics.generators.sprites.TextMotionSprite;
import net.electroland.noho.util.SensorPair;
import net.electroland.noho.util.scheduler.TimedEvent;
import net.electroland.noho.util.scheduler.TimedEventListener;
import net.electroland.noho.weather.TempChecker;
import net.electroland.noho.weather.WeatherChangeListener;
import net.electroland.noho.weather.WeatherChangedEvent;
import net.electroland.noho.weather.WeatherChecker;

/**
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */

@SuppressWarnings("serial")

public class MainFrame extends JFrame implements ImageConsumer, TimedEventListener, WeatherChangeListener {
	
	final static double vadj = 10;
	
	
	TimedEvent sunriseOn = new TimedEvent(6,00,00, this); // on at sunrise-1 based on weather
	TimedEvent middayOff = new TimedEvent(12,00,00, this); // off at 12 PM for sun reasons
	TimedEvent sunsetOn = new TimedEvent(16,00,00, this); // on at sunset-1 based on weather
	TimedEvent nightOff = new TimedEvent(2,00,00, this); // off at 2 AM
	
	TextQueue textQueue = new TextQueue();
	TrafficObserver trafficobserver;
	Compositor compositor;
	AnimationManager animationManager;
	RenderThread render;
	SensorThread northSensor;
	SensorThread southSensor;
	BufferedImage imageBuffer;
	
	WeatherChecker weatherChecker;
	TempChecker tempChecker;
	float lastTemp;
	boolean overTempShutdown;
	
		
	public MainFrame() { 
		super("NoHo");
	}
	
	/**
	 * @param w
	 * @param h
	 */
	public void setup(int w, int h) {
		//System.out.println("setting up");
		setSize(w, h);
		imageBuffer = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
		compositor = new Compositor(w,h);
		compositor.setConsumer(this);
		trafficobserver = new TrafficObserver();
		animationManager = new AnimationManager(w,h, compositor, textQueue, trafficobserver);
		

		// wait 6 secs (for things to get started up) then check weather every half hour
		weatherChecker = new WeatherChecker(6000, 60 * 30 * 1000);
		weatherChecker.addListener(this);
		
		
		// wait 2 secs (for things to get started up) then check artboxtemp every 10 minutes
		tempChecker = new TempChecker(2000, 60 * 10 * 1000);
		tempChecker.addListener(this);
		
		if (!NoHoConfig.TESTING){
			weatherChecker.start(); // never need to stop
			tempChecker.start(); // never need to stop
		} else {
			System.out.println("TESTING MODE, no weather or temperature actions");
		}
	}
	
	public void start() {
		if(render == null) { // don't want to start twice
			render = new RenderThread(NoHoConfig.FRAMERATE);
			render.start();
		}		
		if (northSensor == null){
			System.out.println("starting up northern camera.");
			PresenceDetector ndet = 
				PresenceDetector.createFromFile(new File(NoHoConfig.NORTH_CAMERA_ELV_FNAME));
			AxisCamera ncam = new NoHoNorthCam(160,120, ndet, false);
			
			northSensor = new SensorThread(ndet, ncam, new NoHoConfig().NORTH_SENSOR_PAIRS, 
											animationManager.getSpriteWorld());
			northSensor.start();
		}

		if (southSensor == null){
			System.out.println("starting up southern camera.");
			PresenceDetector sdet = 
				PresenceDetector.createFromFile(new File(NoHoConfig.SOUTH_CAMERA_ELV_FNAME));
			AxisCamera scam = new NoHoSouthCam(160,120, sdet, false);
			
			southSensor = new SensorThread(sdet, scam, new NoHoConfig().SOUTH_SENSOR_PAIRS, 
											animationManager.getSpriteWorld());
			southSensor.start();
		}
		
		trafficobserver.resetAll();
	}
	
	public void stop() {
		if(render == null) {
			Graphics g = getGraphics();
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, getWidth(), getHeight());
		} else {
			render.isRunning = false;
			render = null;
			northSensor.isRunning = false;
			northSensor = null;
			southSensor.isRunning = false;
			southSensor = null;
		}
	}
	
	/**
	 * @param dt
	 * @param curTime
	 */
	public void render(long dt, long curTime) {
		Graphics2D g2d =  imageBuffer.createGraphics();
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		g2d.setColor(Color.BLACK);
		g2d.fillRect(0, 0, imageBuffer.getWidth(), imageBuffer.getHeight());
		
		animationManager.nextFrame(dt, curTime);
		
		g2d = (Graphics2D) getGraphics();
		g2d.drawImage(imageBuffer, 0,0, null);
		
	}

	public void paint(Graphics g) {

	}

	public void renderImage(long dt, long curTime, BufferedImage image) {
		Graphics2D g2 = imageBuffer.createGraphics();
		g2.drawImage(image, 0, 0 ,null);
	}
	
	
	
	
	
	//SENSOR CODE
	
	public class SensorThread extends Thread {

		boolean isRunning = true;
		AxisCamera cam;
		PresenceDetector detector;
		SpriteImageGenerator spriteWorld;
		Vector<SensorPair>sensorPairs;
		
		public SensorThread(PresenceDetector detector,
							AxisCamera cam,
							Vector<SensorPair>sensorPairs,
							SpriteImageGenerator spriteWorld){
			this.detector = detector;
			this.cam = cam;
			this.sensorPairs = sensorPairs;
			this.spriteWorld = spriteWorld;
		}

		public void startNorthernCamSprite(long time, int lane){
			if (NoHoConfig.TESTING){
				System.out.println("NORTH - NEW CAR in lane " + lane + " w/ time " + time);
			}
			
			//register the new car with trafficobserver
			trafficobserver.carDetected(true);

			// TEXT SPRITES
			String tmpText = "Hi!";
			TextMotionSprite sprite = new TextMotionSprite(tmpText.length()*-10,0,Color.WHITE,NoHoConfig.DISPLAYWIDTH+(tmpText.length()*10),0,time,tmpText,spriteWorld);
			spriteWorld.addSprite(sprite).addListener(sprite);


			/** OLD RECT SPRITE CODE

				// RECTANGLE
				LinearMotionRecSprite sprite 
				= new LinearMotionRecSprite(
						2.0 - NoHoConfig.NORTH_INIT_X_OFFSET, 	// init x
						1.0,									// init y
						NoHoConfig.DISPLAYHEIGHT-2.0,			// rect width
						NoHoConfig.DISPLAYHEIGHT-2.0,			// rect height
						Color.RED,								// rect color
						(double)NoHoConfig.DISPLAYWIDTH,		// dest x
						1.0,									// dest y
						time,
						spriteWorld); // time to get there 

				spriteWorld.addSprite(sprite).addListener(sprite);
			 * 
			 */




			
		}

		public void startSouthernSprite(long time, int lane){
			if (NoHoConfig.TESTING){
				System.out.println("SOUTH - NEW CAR in lane " + lane + " w/ time " + time);
			}

			//register the new car with trafficobserver
			trafficobserver.carDetected(false);

			// TEXT SPRITES
			String tmpText = "Bye";
			TextMotionSprite sprite = new TextMotionSprite(NoHoConfig.DISPLAYWIDTH+(tmpText.length()*10),0,Color.WHITE,tmpText.length()*-10,0,time,tmpText,spriteWorld);
			spriteWorld.addSprite(sprite).addListener(sprite);

		}
		
		public void run() {

			cam.start();
			detector.start(); // start detector before cam or else you'll get synchronization problems
			Vector<PolyRegion> regions = detector.getRegions();
			boolean[] triggers = new boolean[regions.size()];
			
			// init to false which should be done by java anyway
			for(int i =0; i < triggers.length; i++) {
				triggers[i] = false;
			}

			while(isRunning) {
				int i = 0;
				for(PolyRegion region : regions) { // check all the regions
					if(region.isTriggered) {  // if triggered now
						if(! triggers[i]) { // but not previously
							
							triggers[i] = true;

							for (SensorPair pair : sensorPairs){
								if (pair.type == NoHoConfig.SOUTH){
									//System.out.println("SOUTH " + region.name + " is triggered");										
								}
								if (pair.type == NoHoConfig.NORTH){
									//System.out.println("NORTH " + region.name + " is triggered");										
								}
								if (region.id == pair.startSensorId){
									if (!pair.waiting){
										pair.startTime = System.currentTimeMillis();
										pair.waiting = true;
									}
								}else if (region.id == pair.endSensorId){									
									if (pair.waiting){
										long time = System.currentTimeMillis() - pair.startTime;
										if (time <= pair.threshold){
											
											time *= pair.tmultiplier;
											time = time < NoHoConfig.UPPER_TIME_LIMIT ? time : NoHoConfig.UPPER_TIME_LIMIT;
											time = time > NoHoConfig.LOWER_TIME_LIMIT ? time : NoHoConfig.LOWER_TIME_LIMIT;
											
											switch (pair.type){
											case(NoHoConfig.NORTH):
												startNorthernCamSprite(time, pair.id);
												break;
											case(NoHoConfig.SOUTH):
												startSouthernSprite(time, pair.id);
												break;
											}											
										}else{
											if (NoHoConfig.TESTING){
												System.out.println("ignoring vehicle at speed " + time);
											}
										}
										pair.waiting = false;
									}
								}
							}
						}
					} else { // if not triggered now
						if(triggers[i]) { // but was previosly
//							System.out.println(region.name + " is no longer triggered");
							triggers[i] = false;
						}					
					}
					i++;

				}
				// sleep a bit so we don't hammer the processor
				try { sleep(100); } catch (InterruptedException e) {};
			}
			cam.stopRunning();
			detector.stopRunning();
		}		
	}
	

	
	
	// MAIN RENDERING THREAD
	
	
	public class RenderThread extends Thread {
		boolean isRunning;
		
		long  ticksPerFrame;
		
		public RenderThread(float fps) {
			ticksPerFrame = (long) (1000.0f / fps); 
			System.out.println("Rendering at " + fps + " fps");
			isRunning = true;
		}
		
		
		public void run() {
			boolean notWarned = false;
			
			long startTime;
			long lastTime;
			long dTime;
			
			lastTime = System.currentTimeMillis();
			while(isRunning) {
				startTime = System.currentTimeMillis();
				dTime = startTime - lastTime;
				render(dTime, startTime);
				
				trafficobserver.process();

				dTime = startTime + ticksPerFrame - System.currentTimeMillis();
				
				if(dTime > 0) {
					notWarned = true;
					try {
						sleep(dTime);
					} catch (InterruptedException e) {
					}
					
				} else {
					if(notWarned) {
						System.err.println("Warning: framerate falling behind");
						notWarned = false;
					}
				}
				lastTime = startTime;
				
			}
			Graphics g = getGraphics();
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, getWidth(), getHeight());
			
		}
	}
	
	
	
	
	
	//
	//SCHEDULING
	//
	
	public void timedEvent(TimedEvent event) {
		if (!NoHoConfig.TESTING){
			if(event == sunriseOn) {
				start();
			} else if (event == middayOff) {
				stop();
			} else if (event == sunsetOn){
				start();
			} else if (event == nightOff){
				stop();
			}
		}
	}

	public boolean shouldBeRunning() {
			
		/*
		System.out.println("On at " + sunriseOn.dateIterator);
		System.out.println("Off at " + middayOff.dateIterator);
		System.out.println("On at " + sunsetOn.dateIterator);
		System.out.println("Off at " +nightOff.dateIterator);
		*/
	
		if (middayOff.dateIterator.current().before(sunriseOn.dateIterator.current()) == true){
			System.out.println("Should be on in morning");
			return true;
		} else if (nightOff.dateIterator.current().before(sunsetOn.dateIterator.current()) == true){
			System.out.println("Should be on in evening");
			return true;
		} else {
			if (sunsetOn.dateIterator.current().before(middayOff.dateIterator.current()) == true){
				System.out.println("Not running in afternoon.  Next activation is at Sunset");
			} else {
				System.out.println("Not running in wee hours.  Next activation is at Sunrise");
			}
			return false;
		}
	}
	
	public boolean isLateNight() {
		if (sunriseOn.dateIterator.current().before(nightOff.dateIterator.current()) == true){
			return false;
		} else {
			System.out.println("But it's the middle of the night!!");
			return true;
		}
	}
	
	
	
	
	
	
	//
	//CONDITIONAL OPERATIONS, WEATHER & TEMP
	//
	
	public void weatherChanged(WeatherChangedEvent wce) {
		//disable all this stuff for alwaysrun testing
		if (!NoHoConfig.TESTING){
			
			if(wce.hasSunriseChanged()) {
				Calendar sunrise = wce.getRecord().getSunrise();
				int h = sunrise.get(Calendar.HOUR_OF_DAY);
				int m = sunrise.get(Calendar.MINUTE);
				int s = sunrise.get(Calendar.SECOND);
				System.out.println("Sunrise at " + h + ":" + m + ":" + s);
				sunriseOn.reschedule(h-1, m, s); // turn off an hour before sunrise
			}
			if(wce.hasSunsetChanged()) {
				Calendar sunset = wce.getRecord().getSunset();
				int h = sunset.get(Calendar.HOUR_OF_DAY);
				int m = sunset.get(Calendar.MINUTE);
				int s = sunset.get(Calendar.SECOND);
				System.out.println("Sunset at " + h + ":" + m + ":" + s);
				sunsetOn.reschedule(h - 1, m, s); // turn on 1 hour before sunset
			}

			System.out.println("CONDITION = " + wce.getRecord().getCondition());
			System.out.println("VISIBILITY = " + wce.getRecord().getVisibility());
			System.out.println("OUTSIDE TEMP = " + wce.getRecord().getOutsideTemperature());


			// check for midday haze, etc
			if (sunsetOn.dateIterator.current().before(middayOff.dateIterator.current()) == true){

				// if conditions are lower than 29, ie mostly cloudy or worse, and vis is less thatn 10 miles, startup
				if (wce.getRecord().getCondition() < NoHoConfig.LOWCONDITION && wce.getRecord().getVisibility() < NoHoConfig.LOWVISIBILITY) {
					if (lastTemp < NoHoConfig.MAXTEMP){
						if (render == null) {
							System.out.println("Starting in the afternoon; weather condition = " + wce.getRecord().getCondition() + ", visibility = " + wce.getRecord().getVisibility() + ", temp = " + lastTemp);
							start();
						}
					}
				} else {
					if (render != null) {
						System.out.println("Stopping in the afternoon due to weather condition at temp " + lastTemp);
						stop();
					}
				}
			}
		}

	}


	// called by TempChecker
	public void tempUpdate(float temp){
		//disable all this stuff for alwaysrun testing
		if (!NoHoConfig.TESTING){
			lastTemp = temp;
			if (temp > NoHoConfig.MAXTEMP && !overTempShutdown){
				System.out.println("WeatherGoose reports: " + temp + ", above maxtemp: " + NoHoConfig.MAXTEMP);
				System.out.println("Stopping");
				overTempShutdown = true;
				stop();
			} else if (overTempShutdown) {
				if (shouldBeRunning()){
					System.out.println("WeatherGoose reports: " + temp);
					System.out.println("Starting");
					overTempShutdown = false;
					start();
				}
			} else {
				System.out.println("WeatherGoose reports: " + temp + " = normal");
			}
		}
	}
	
	
	
	
	
	
	
	// MAIN
	
	public static void main(String[] args) {
		MainFrame frame = new MainFrame();
		frame.setUndecorated(true);
		frame.setup(NoHoConfig.DISPLAYWIDTH,NoHoConfig.DISPLAYHEIGHT);
		frame.setVisible(true);
		
		//added by DS for simple exiting
		frame.addMouseListener(new MouseListener() {
			public void mouseClicked(MouseEvent e) {
				System.exit(0);
			}
			public void mousePressed(MouseEvent e) {
			}
			public void mouseReleased(MouseEvent e) {
			}
			public void mouseEntered(MouseEvent e) {
			}
			public void mouseExited(MouseEvent e) {
			}
		});

		//disable all this stuff for alwaysrun testing
		if (!NoHoConfig.TESTING){
			if(frame.shouldBeRunning()) {
				frame.start();
			} else {
				frame.stop();
			}
		} else {
			System.out.println("TESTING MODE, render will run continuously");
			frame.start();
		}


	}
	
	


	


}
