package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

// # Single tile based sprite for LilyPad style activation

public class Pad extends Sprite {
	
	private int duration, minValue, maxValue, value, holdTime;
	private boolean fadeIn, fadeOut, hold;
	private long startTime;
	private long padStartTime;
	private int timeOut;
	public boolean activated = false;

	public Pad(int id, Raster raster, int x, int y, SoundManager sm, int minValue, int maxValue, int duration) {
		super(id, raster, x, y, sm);
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.duration = duration;
		this.value = this.minValue;
		holdTime = 2000 + (int)(Math.random()*4000);
		fadeIn = true;
		fadeOut = false;
		hold = false;
		timeOut = 90000 + (int)(Math.random()*30000);
		startTime = System.currentTimeMillis();
		padStartTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(fadeIn){
			value = minValue + (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (maxValue-minValue));
			if(value >= maxValue){
				value = maxValue;
				fadeIn = false;
				//fadeOut = true;
				startTime = System.currentTimeMillis();
				hold = true;
			}
		} else if(fadeOut){
			value = maxValue - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (maxValue-minValue));
			if(value <= minValue){
				if(System.currentTimeMillis() - padStartTime > timeOut){
					die();
				}
				value = minValue;
				fadeIn = true;
				fadeOut = false;
				startTime = System.currentTimeMillis();
			}
		} else if(hold){
			if((System.currentTimeMillis() - startTime) > holdTime){
				hold = false;
				fadeOut = true;
				startTime = System.currentTimeMillis();
			}
		}
		
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			c.fill(value,0,0);
			c.noStroke();
			c.rect(x*tileSize, y*tileSize, tileSize, tileSize);	// single tile sized square
			c.popMatrix();
		}
	}
	
	public void fadeOut(int duration){
		this.duration = duration;		// set fade speed to whatever to match sound
		activated = true;
		value = 255;					// start at full brightness
		timeOut = 0;					// die when this has faded out
		startTime = System.currentTimeMillis();	// start now
	}

}
