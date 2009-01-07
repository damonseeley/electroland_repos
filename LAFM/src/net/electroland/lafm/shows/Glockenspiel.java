package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class Glockenspiel extends ShowThread {
	
	private int hour, minute, sec, fadeSpeed, brightness;
	
	public Glockenspiel(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, int hour, int minute, int sec, int fadeSpeed, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
		this.hour = hour;
		this.minute = minute;
		this.sec = sec;
		this.fadeSpeed = fadeSpeed;
		this.brightness = 255;
		if(soundManager != null){
			soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, "SolidColor");
		}
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

		boolean[] channelsInUse = new boolean[6];		// null array of sound channels
		for(int n=0; n<channelsInUse.length; n++){
			channelsInUse[n] = false;
		}
		if(soundManager != null){
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				DMXLightingFixture flower = i.next();
				channelsInUse[flower.getSoundChannel()] = true;
			}
			for(int n=0; n<channelsInUse.length; n++){
				if(channelsInUse[n] != false){
					soundManager.playSimpleSound(soundFile, n, 1.0f, ID);
				}
			}
		}
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
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
