package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

// # Single tile based sprite for LilyPad style activation

public class Pad extends Sprite {
	
	private int duration, minValue, maxValue, value;
	private boolean fadeIn, fadeOut;
	private long startTime;

	public Pad(Raster raster, int x, int y, int minValue, int maxValue, int duration) {
		super(raster, x, y);
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.duration = duration;
		this.value = this.minValue;
		fadeIn = true;
		fadeOut = false;
		startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(fadeIn){
			value = minValue + (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (maxValue-minValue));
			if(value >= maxValue){
				value = maxValue;
				fadeIn = false;
				fadeOut = true;
				startTime = System.currentTimeMillis();
			}
		} else if(fadeOut){
			value = maxValue - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (maxValue-minValue));
			if(value <= minValue){
				value = minValue;
				fadeIn = true;
				fadeOut = false;
				startTime = System.currentTimeMillis();
			}
		}		
		
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			c.fill(value,0,0);
			c.noStroke();
			c.rect((x*tileSize)-1, (y*tileSize)-1, tileSize, tileSize);	// single tile sized square
			c.popMatrix();
		}
	}

}
