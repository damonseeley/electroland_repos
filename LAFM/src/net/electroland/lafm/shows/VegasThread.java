package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class VegasThread extends ShowThread {
	
	ColorScheme spectrum;
	float speed;
	int age = 0;

	public VegasThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		if(soundManager != null){
			soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
		}
	}
	
	public VegasThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		if(soundManager != null){
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				DMXLightingFixture flower = i.next();
				soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
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
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		if(age > speed){
			raster.background(0);
			raster.noStroke();
			for(int i=0; i<25; i++){
				float[] color = spectrum.getColor((float)Math.random());
				raster.fill(color[0], color[1], color[2]);
				raster.rect((float)(Math.random()*255), (float)(Math.random()*255), 50, 50);
			}
			age = 0;
		} else {
			age++;
		}
		raster.endDraw();
	}

}
