package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class Glockenspiel extends ShowThread {
	
	private int fadeSpeed, brightness;
	private float red, green, blue;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int startDelay, delayCount;
	
	public Glockenspiel(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, int red, int green, int blue, int fadeSpeed,
			String soundFile, Properties physicalProps, int startDelay) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
		this.red = (red/255.0f);
		this.green = (green/255.0f);
		this.blue = (blue/255.0f);
		this.fadeSpeed = fadeSpeed;
		this.brightness = 255;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		delayCount = 0;
	}

	public Glockenspiel(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, int red, int green, int blue, int fadeSpeed,
			String soundFile, Properties physicalProps,  int startDelay) {
		super(flowers, soundManager, lifespan, fps, raster, ID, priority);
		this.red = (red/255.0f);
		this.green = (green/255.0f);
		this.blue = (blue/255.0f);
		this.fadeSpeed = fadeSpeed;
		this.brightness = 255;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		delayCount = 0;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		if(delayCount >= startDelay){
			if(startSound){
				super.playSound(soundFile, physicalProps);
				startSound = false;
			}
			
			raster.colorMode(PConstants.RGB, 255, 255, 255);
			raster.beginDraw();
			raster.background(red*brightness, green*brightness, blue*brightness);
			raster.endDraw();
			if(brightness > 0){
				brightness -= fadeSpeed;
			} else {
				cleanStop();
			}
		} else {
			delayCount++;
			super.resetLifespan();
		}
	}

}
