package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class KnockoutThread extends ShowThread implements SensorListener{
	
	private ColorScheme spectrum;
	private ConcurrentHashMap<Integer,Box> boxes;
	private int[] inactiveBoxes;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int boxCount = 0;
	private int age = 0;
	private int fadeSpeed;
	private float minColorPoint, maxColorPoint;
	private float colorShiftSpeed, spectrumShiftSpeed;
	private boolean spectrumDirection;
	private boolean knockOut = false;
	private int duration;	// counting frames before fading out
	private float gain;

	public KnockoutThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int rowCount, int fadeSpeed, float minColorPoint,
			float maxColorPoint, float colorShiftSpeed, float spectrumShiftSpeed,
			String soundFile, Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.minColorPoint = minColorPoint;
		this.maxColorPoint = maxColorPoint;
		this.colorShiftSpeed = colorShiftSpeed;
		this.spectrumShiftSpeed = spectrumShiftSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.gain = gain;
		this.spectrumDirection = true;
		boxes = new ConcurrentHashMap<Integer,Box>();
		for(int x=0; x<rowCount; x++){
			for(int y=0; y<rowCount; y++){
				boxes.put(boxCount, new Box(boxCount, x, y, raster.width/rowCount, raster.height/rowCount));
				boxCount++;
			}
		}
		inactiveBoxes = new int[boxCount];
		for(int i=0; i<boxCount; i++){
			inactiveBoxes[i] = i;
		}
		duration = ((int)(lifespan/1000.0f)*fps) - ((rowCount*rowCount) + 100/fadeSpeed);
		this.startSound = true;
	}
	
	public KnockoutThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int rowCount, int fadeSpeed, float minColorPoint,
			float maxColorPoint, float colorShiftSpeed, float spectrumShiftSpeed,
			String soundFile, Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.minColorPoint = minColorPoint;
		this.maxColorPoint = maxColorPoint;
		this.colorShiftSpeed = colorShiftSpeed;
		this.spectrumShiftSpeed = spectrumShiftSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.gain = gain;
		this.spectrumDirection = true;
		boxes = new ConcurrentHashMap<Integer,Box>();
		for(int x=0; x<rowCount; x++){
			for(int y=0; y<rowCount; y++){
				boxes.put(boxCount, new Box(boxCount, x, y, raster.width/rowCount, raster.height/rowCount));
				boxCount++;
			}
		}
		inactiveBoxes = new int[boxCount];
		for(int i=0; i<=boxCount; i++){
			inactiveBoxes[i] = i;
		}
		duration = ((int)(lifespan/1000.0f)*fps) - ((rowCount*rowCount) + 100/fadeSpeed);
		this.startSound = true;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}
	
	public void removeBox(){
		int luckynumber = (int)(Math.random()*(inactiveBoxes.length-1));
		//System.out.println(luckynumber +" "+ inactiveBoxes[luckynumber] +" inactive: "+ inactiveBoxes.length);
		boxes.get(inactiveBoxes[luckynumber]).beginDecay();
		int[] newboxes = new int[inactiveBoxes.length-1];
		System.arraycopy(inactiveBoxes, 0, newboxes, 0, luckynumber);											// add everything before number
		System.arraycopy(inactiveBoxes, luckynumber+1, newboxes, luckynumber, inactiveBoxes.length-luckynumber-1);	// add everything after number
		inactiveBoxes = newboxes;
	}

	@Override
	public void doWork(PGraphics raster) {
		if(startSound){
			super.playSound(soundFile, gain, physicalProps);
			startSound = false;
		}
		
		if(maxColorPoint >= 1){						// color range moving up or down spectrum
			spectrumDirection = false;
			maxColorPoint = 1;
		} else if(minColorPoint <= 0){
			spectrumDirection = true;
			minColorPoint = 0;
		}
		
		if(spectrumDirection){						// color range movement
			maxColorPoint += spectrumShiftSpeed;
			minColorPoint += spectrumShiftSpeed;
		} else {
			maxColorPoint -= spectrumShiftSpeed;
			minColorPoint -= spectrumShiftSpeed;
		}

		// draw all the boxes
		raster.beginDraw();
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.noStroke();
		Iterator<Box> i = boxes.values().iterator();
		while (i.hasNext()){
			Box b = i.next();
			b.draw(raster);
		}
		raster.endDraw();
		
		if(age > duration){
			knockOut = true;
		}
		age++;
		
		if(knockOut && age > 60){	// start the decay of a random box (after minimum time on)
			if(inactiveBoxes.length > 0){
				removeBox();
				removeBox();
			}
		}
		
		// end the show when all boxes are gone
		if(boxes.size() == 0){
			cleanStop();
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			knockOut = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// no reactivation for this one
		}
	}
	
	
	
	public class Box{
		private int id, x, y, width, height;
		private float red, green, blue;	// normalized color values
		private int brightness;
		private float[] color;
		private float colorPoint;
		private boolean fadeOut = false;
		private boolean colorDirection;
		
		public Box(int id, int x, int y, int width, int height){
			this.id = id;
			this.x = x*width;
			this.y = y*height;
			this.width = width;
			this.height = height;
			this.colorPoint = (float)(Math.random()*maxColorPoint) - minColorPoint;
			//System.out.println(maxColorPoint +" "+ minColorPoint +" "+ colorPoint);
			color = spectrum.getColor(colorPoint);
			red = color[0]/255.0f;
			green = color[1]/255.0f;
			blue = color[2]/255.0f;
			brightness = 255;
			if(Math.random() > 0.5){
				colorDirection = true; 
			} else {
				colorDirection = false; 
			}
		}
		
		public void beginDecay(){
			fadeOut = true;
		}
		
		public void draw(PGraphics raster){						
			if(colorDirection){							// color movement
				colorPoint += colorShiftSpeed;
			} else {
				colorPoint -= colorShiftSpeed;
			}
			
			if(colorPoint >= maxColorPoint){			// color moving up or down range
				colorDirection = false;
				colorPoint = maxColorPoint;
			} else if(colorPoint <= minColorPoint){
				colorDirection = true;
				colorPoint = minColorPoint;
			}
			
			color = spectrum.getColor(colorPoint);
			red = color[0]/255.0f;
			green = color[1]/255.0f;
			blue = color[2]/255.0f;
			
			raster.fill(red*brightness, green*brightness, blue*brightness);
			raster.rect(x,y,width,height);
			
			if(fadeOut){
				if(brightness > 0){
					brightness -= fadeSpeed;
				} else {
					boxes.remove(id);
				}
			}
		}
	}

}
