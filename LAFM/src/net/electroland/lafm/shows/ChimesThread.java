package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ChimesThread extends ShowThread {
	
	/*
	 * This show goes off each hour and issues visual/audible chimes
	 * to count what hour it is.
	 */

	private int hour;
	private int brightness, alpha, fadeSpeed;
	private float red, green, blue;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	
	public ChimesThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int hour,	int fadeSpeed,
			int red, int green, int blue, String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.hour = hour;
		this.red = (red/255.0f);
		this.green = (green/255.0f);
		this.blue = (blue/255.0f);
		this.fadeSpeed = fadeSpeed;
		brightness = 255;
		alpha = 100;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}
	
	public ChimesThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int hour,	int fadeSpeed,
			int red, int green, int blue, String soundFile, Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.hour = hour;
		this.red = (red/255.0f);
		this.green = (green/255.0f);
		this.blue = (blue/255.0f);
		this.fadeSpeed = fadeSpeed;
		brightness = 255;
		alpha = 100;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
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
		raster.background(0,0,0);
		raster.noStroke();
		for(int i=0; i<25; i++){
			raster.fill(red*brightness, green*brightness, blue*brightness);
			raster.rect((float)(Math.random()*255), (float)(Math.random()*255), 50, 50);
		}
		raster.fill(red*255, green*255, blue*255, alpha);
		raster.rect(0,0,256,256);
		raster.endDraw();
		if(alpha > 0){
			alpha -= fadeSpeed;
		} else {
			if(brightness > 0){				// fade out
				brightness -= fadeSpeed;
			} else {
				cleanStop();
			}
		}		
	}

}
