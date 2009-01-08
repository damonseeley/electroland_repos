package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class VegasThread extends ShowThread implements SensorListener{
	
	ColorScheme spectrum;
	float speed;
	int age = 0;
	int delay = 0;
	int alpha = 0;
	int fadeSpeed = 3;
	private boolean startSound, fadeOut;
	private String soundFile;
	private int duration;	// counting frames before fading out

	public VegasThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.soundFile = soundFile;
		startSound = true;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
	}
	
	public VegasThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.soundFile = soundFile;
		startSound = true;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
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
			super.playSound(soundFile);
			startSound = false;
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		if(delay > speed){
			raster.background(0);
			raster.noStroke();
			for(int i=0; i<25; i++){
				float[] color = spectrum.getColor((float)Math.random());
				raster.fill(color[0], color[1], color[2]);
				raster.rect((float)(Math.random()*255), (float)(Math.random()*255), 50, 50);
			}
			delay = 0;
		} else {
			delay++;
		}
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(128,128,raster.width,raster.height);
			} else {
				cleanStop();
			}
		}
		raster.endDraw();
		age++;
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			// slow down when sensor triggered off
			fadeOut = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			fadeOut = false;
			alpha = 0;
			age = 0;
		}
	}

}
