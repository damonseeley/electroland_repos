package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class BullsEye extends Sprite {
	
	private Person person;
	private int ringCount, fadeSpeed, alpha, timeOut;
	private long[] startTime;
	private long fadeStartTime, trueStartTime;
	private int[] brightness;
	private boolean fadeOut;

	public BullsEye(int id, Raster raster, float x, float y, SoundManager sm, Person person, int ringCount, int fadeSpeed) {
		super(id, raster, x, y, sm);
		this.person = person;					// tracks person for persistent animation
		this.ringCount = ringCount;				// number of "rings" around the bulls eye
		this.fadeSpeed = fadeSpeed;				// speed in milliseconds to go from 0-255 brightness
		brightness = new int[ringCount];		// contains brightness fluctuations
		startTime = new long[ringCount];
		alpha = 255;
		for(int i=0; i<ringCount; i++){
			brightness[i] = (i/ringCount)*255;
			startTime[i] = System.currentTimeMillis() - (long)((i/(float)ringCount)*fadeSpeed);	// offset equally
		}
		timeOut = 5000;
		trueStartTime = System.currentTimeMillis();
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("test2"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);
			c.noStroke();
			for(int i=0; i<ringCount; i++){
				if(System.currentTimeMillis() - startTime[i] > fadeSpeed){
					startTime[i] = System.currentTimeMillis();
				}
				brightness[i] = (int)(Math.sin(((System.currentTimeMillis() - startTime[i]) / (float)fadeSpeed)*Math.PI)*255);
				System.out.println("bullseye ring"+i+" "+brightness[i]);
				c.fill(brightness[i],0,0,alpha);
				c.rect(x, y, ((ringCount-i)*(tileSize*2) - tileSize), ((ringCount-i)*(tileSize*2) - tileSize));
			}
			c.popMatrix();
		}
		if(((person != null && person.isDead()) || System.currentTimeMillis() - trueStartTime > timeOut) && !fadeOut){
			fadeStartTime = System.currentTimeMillis();
			fadeOut = true;
		}
		if(fadeOut){
			alpha = 255 - (int)(((System.currentTimeMillis() - fadeStartTime) / 1000) * 255);
			if(alpha <= 0){
				die();
			}
		}
	}

}
