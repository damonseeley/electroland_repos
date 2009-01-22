package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class KnockoutThread extends ShowThread {
	
	private ColorScheme spectrum;
	private ConcurrentHashMap<Integer,Box> boxes;
	private int[] inactiveBoxes;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int boxCount = 0;
	private int fadeSpeed;

	public KnockoutThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int rowCount, int fadeSpeed, String soundFile,
			Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
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
		this.startSound = true;
	}
	
	public KnockoutThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int rowCount, int fadeSpeed, String soundFile,
			Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
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
		this.startSound = true;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		if(startSound){
			super.playSound(soundFile, physicalProps);
			startSound = false;
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
		
		// start the decay of a random box
		if(inactiveBoxes.length > 0){
			int luckynumber = (int)(Math.random()*(inactiveBoxes.length-1));
			System.out.println(luckynumber +" "+ inactiveBoxes[luckynumber] +" inactive: "+ inactiveBoxes.length);
			boxes.get(inactiveBoxes[luckynumber]).beginDecay();
			int[] newboxes = new int[inactiveBoxes.length-1];
			System.arraycopy(inactiveBoxes, 0, newboxes, 0, luckynumber);											// add everything before number
			System.arraycopy(inactiveBoxes, luckynumber+1, newboxes, luckynumber, inactiveBoxes.length-luckynumber-1);	// add everything after number
			inactiveBoxes = newboxes;
		}
		
		// end the show when all boxes are gone
		if(boxes.size() == 0){
			cleanStop();
		}
	}
	
	
	
	public class Box{
		private int id, x, y, width, height;
		private float red, green, blue;	// normalized color values
		private int brightness;
		private float[] color;
		private boolean fadeOut = false;
		
		public Box(int id, int x, int y, int width, int height){
			this.id = id;
			this.x = x*width;
			this.y = y*height;
			this.width = width;
			this.height = height;
			color = spectrum.getColor((float)Math.random());
			red = color[0]/255.0f;
			green = color[1]/255.0f;
			blue = color[2]/255.0f;
			brightness = 255;
		}
		
		public void beginDecay(){
			fadeOut = true;
		}
		
		public void draw(PGraphics raster){
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
