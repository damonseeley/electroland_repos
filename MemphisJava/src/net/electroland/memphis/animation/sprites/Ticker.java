package net.electroland.memphis.animation.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;

public class Ticker extends Sprite {
	
	/*
	 * Ticker.java
	 * Just like Shooter, but slow, soft, sideways, and looping.
	 */
	
	private PImage image;
	private int duration, fadeDuration;
	private long startTime, fadeStartTime;
	private boolean switchDirection, fadeOut;
	private float r, g, b, alpha;

	public Ticker(int id, Raster raster, float x, float y, PImage image, float width, float height, int duration, boolean switchDirection) {
		super(id, raster, x, y);
		this.width = width;
		this.height = height;
		this.image = image;
		this.duration = duration;
		this.switchDirection = switchDirection;
		alpha = 255;
		r = g = b = 255;	// default white
		fadeDuration = 250;
		fadeOut = false;
		startTime = System.currentTimeMillis();
	}

	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			c.pushMatrix();
			c.tint(r,g,b,alpha);
			
			if(!fadeOut){
				if(switchDirection){
					y = c.height - (((System.currentTimeMillis() - startTime) / (float)duration) * (c.height+height));
					if(y <= 0-height){
						die();
					}
					c.translate(x, y);
					c.rotate((float)Math.PI);	// flip it
					c.image(image, 0,  0-height, width, height);
				} else {
					y = (((System.currentTimeMillis() - startTime) / (float)duration) * (c.height+height));
					if(y >= c.height+height){
						die();
					}
					c.image(image, x, y-height, width, height);
				}
			} else {
				alpha = 255 - (((System.currentTimeMillis() - fadeStartTime) / fadeDuration) * 255);
				if(alpha <= 0){
					die();
				}
			}
			c.tint(255,255,255,255);	// set back to opaque, since processing has a bug with tint
			c.popMatrix();
		}
	}

	public void setColor(float r, float g, float b){
		this.r = r;
		this.g = g;
		this.b = b;
	}

}
