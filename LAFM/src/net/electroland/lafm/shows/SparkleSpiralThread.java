package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class SparkleSpiralThread extends ShowThread implements SensorListener{
	
	ColorScheme spectrum;
	float sparkleSpeed, spiralSpeed;
	float[] color;
	float rotation = 0;
	int sparkleDelay = 0;
	int spiralDelay = 0;
	int sparkleCount = 0;
	int spriteWidth = 50;
	boolean fadeIn, fadeOut, interactive;
	
	public SparkleSpiralThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float sparkleSpeed,
			float spiralSpeed, boolean interactive, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.sparkleSpeed = sparkleSpeed;
		this.spiralSpeed = spiralSpeed;
		this.interactive = interactive;
		fadeIn = true;
		fadeOut = false;
		if(soundManager != null){
			soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
		}
	}

	public SparkleSpiralThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float sparkleSpeed,
			float spiralSpeed, boolean interactive, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.sparkleSpeed = sparkleSpeed;
		this.spiralSpeed = spiralSpeed;
		this.interactive = interactive;
		fadeIn = true;
		fadeOut = false;

		boolean[] channelsInUse = new boolean[6];		// null array of sound channels
		for(int n=0; n<channelsInUse.length; n++){
			channelsInUse[n] = false;
		}
		if(soundManager != null){
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				DMXLightingFixture flower = i.next();
				channelsInUse[flower.getSoundChannel()-1] = true;
			}
			for(int n=0; n<channelsInUse.length; n++){
				if(channelsInUse[n] != false){
					soundManager.playSimpleSound(soundFile, n+1, 1.0f, ID);
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
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.background(0);
		raster.noStroke();
		raster.translate(128, 128);
		
		for(int i=0; i<sparkleCount; i++){
			raster.pushMatrix();
			color = spectrum.getColor((float)Math.random());
			raster.fill(color[0], color[1], color[2]);
			if(!fadeOut){						// draw from outside --> in
				if(i < 16){						// draw 16 rectangles on the outside
					rotation = (360/16.0f)*i;
					raster.rotate((float)(rotation * Math.PI/180));
					raster.rect(0,100,spriteWidth,spriteWidth);
				} else if(i >= 16 && i < 24){	// draw 8 rectangles on the inside
					rotation = (360/8.0f)*i;
					raster.rotate((float)(rotation * Math.PI/180));
					raster.rect(0,50,spriteWidth,spriteWidth);
				} else {						// draw 1 rectangle in the center
					raster.rect(0,0,spriteWidth,spriteWidth);
				}
			} else {							// draw from inside --> out
				if(i < 1){						// draw 1 rectangle in the center
					raster.rect(0,0,spriteWidth,spriteWidth);
				} if(i >= 1 && i < 9){			// draw 8 rectangles on the inside
					rotation = (360/8.0f)*(24-i);
					raster.rotate((float)(rotation * Math.PI/180));
					raster.rect(0,50,spriteWidth,spriteWidth);
				} else {						// draw 16 rectangles on the outside
					rotation = (360/16.0f)*(24-i);
					raster.rotate((float)(rotation * Math.PI/180));
					raster.rect(0,100,spriteWidth,spriteWidth);
				}
			}
			raster.popMatrix();
		}
		raster.endDraw();
		
		if(fadeIn){
			if(spiralDelay < spiralSpeed){	// if delay not long enough...
				spiralDelay++;				// wait more
			} else {						// delay matches refresh speed
				if(sparkleCount < 25){		// less than total number of sparkles
					sparkleCount++;			// add another sparkle
				} else {
					if(!interactive){
						fadeIn = false;
						fadeOut = true;
					}
				}
				spiralDelay = 0;
			}
		} else if(fadeOut){
			if(spiralDelay < spiralSpeed){	// if delay not long enough...
				spiralDelay++;				// wait more
			} else {						// delay matches refresh speed
				if(sparkleCount >0){		// more than 0
					sparkleCount--;			// remove another sparkle
				} else {
					cleanStop();
				}
				spiralDelay = 0;
			}
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if(interactive){
			if (this.getFlowers().contains(eventFixture) && !isOn){
				//this.cleanStop();
				// make the sparkles spiral out to black
			} else if(this.getFlowers().contains(eventFixture) && isOn){
				// reactivate
				// if sparkles are spiraling out, make them spiral back in
			}
		}
	}

}
