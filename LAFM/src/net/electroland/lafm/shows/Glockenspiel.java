package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class Glockenspiel extends ShowThread {
	
	private int hour, minute, sec, fadeSpeed, brightness;
	private boolean startSound;
	private String soundFile;
	
	public Glockenspiel(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, int hour, int minute, int sec, int fadeSpeed, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
		this.hour = hour;
		this.minute = minute;
		this.sec = sec;
		this.fadeSpeed = fadeSpeed;
		this.brightness = 255;
		this.soundFile = soundFile;
		startSound = true;
	}

	public Glockenspiel(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, int hour, int minute, int sec, int fadeSpeed, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, priority);
		this.hour = hour;
		this.minute = minute;
		this.sec = sec;
		this.fadeSpeed = fadeSpeed;
		this.brightness = 255;
		this.soundFile = soundFile;
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
			super.playSound(soundFile);
			startSound = false;
		}
		
		raster.colorMode(PConstants.HSB, 60, 255, 255);
		raster.beginDraw();
		raster.background(minute, 255, brightness);
		raster.endDraw();
		if(brightness > 0){
			brightness -= fadeSpeed;
		} else {
			cleanStop();
		}
	}

}
