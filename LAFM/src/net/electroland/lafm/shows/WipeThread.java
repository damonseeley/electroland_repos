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

public class WipeThread extends ShowThread implements SensorListener{
	
	private int alpha, barWidth;
	private ColorScheme spectrum;
	private int fadeSpeed, age, duration;
	private String soundFile;
	private Properties physicalProps;
	private boolean startSound, fadeOut;
	private ConcurrentHashMap<Integer,Bar> bars;
	private int barCount = 0;

	public WipeThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int wipeSpeed, int fadeSpeed,
			int numberBars, String soundFile, Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		alpha = 0;
		age = 0;
		barWidth = raster.width/12;
		duration = (lifespan*fps) - (100/fadeSpeed);
		bars = new ConcurrentHashMap<Integer,Bar>();
		for(int i=0; i<numberBars; i++){
			bars.put(barCount, new Bar(raster, wipeSpeed));
			barCount++;
		}
		startSound = true;
	}
	
	public WipeThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, int wipeSpeed, int fadeSpeed,
			int numberBars, String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		alpha = 0;
		age = 0;
		barWidth = raster.width/12;
		duration = (lifespan*fps) - (100/fadeSpeed);
		bars = new ConcurrentHashMap<Integer,Bar>();
		for(int i=0; i<numberBars; i++){
			bars.put(barCount, new Bar(raster, wipeSpeed));
			barCount++;
		}
		startSound = true;
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
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.beginDraw();
		raster.noStroke();
		Iterator<Bar> i = bars.values().iterator();
		while (i.hasNext()){
			Bar b = i.next();
			b.draw(raster);
		}
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,raster.width,raster.height);
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
			} 
			if(alpha >= 100){
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
				cleanStop();
			}
		}
		age++;
		raster.endDraw();
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			fadeOut = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			fadeOut = false;
			alpha = 0;
			age = 0;
		}
	}
	
	
	
	
	public class Bar{
		private int red, green, blue;
		private int y, rotation;
		private int wipeSpeed, suggestedWipeSpeed;
		
		public Bar(PGraphics raster, int wipeSpeed){
			this.suggestedWipeSpeed = wipeSpeed;
			this.wipeSpeed = (int)(Math.random()*wipeSpeed) + wipeSpeed/2;
			float[] color = spectrum.getColor((float)Math.random());
			red = (int)color[0];
			green = (int)color[1];
			blue = (int)color[2];
			y = 0;
			rotation = (int)(Math.random()*360);
		}
		
		public void reset(){
			wipeSpeed = (int)(Math.random()*suggestedWipeSpeed) + suggestedWipeSpeed/2;
			float[] color = spectrum.getColor((float)Math.random());
			red = (int)color[0];
			green = (int)color[1];
			blue = (int)color[2];
			y = 0;
			rotation = (int)(Math.random()*360);
		}
		
		public void draw(PGraphics raster){
			if(y < raster.height){
				y += wipeSpeed;
			} else {
				reset();
			}
			raster.pushMatrix();
			raster.translate(raster.width/2, raster.height/2);
			raster.rotate((float)(rotation * Math.PI/180));
			raster.translate(-raster.width/2, -raster.height/2);
			raster.fill(red, green, blue);
			raster.rect(0,y,raster.width,barWidth);
			raster.popMatrix();
		}
	}

}
