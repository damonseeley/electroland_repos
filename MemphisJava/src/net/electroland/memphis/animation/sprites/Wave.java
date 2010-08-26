package net.electroland.memphis.animation.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;

public class Wave extends Sprite {
	
	private PImage image;
	private int duration;
	private long startTime;
	private boolean switchDirection;
	private float r, g, b, alpha;

	public Wave(int id, Raster raster, float x, float y, PImage image, float width, float height, int duration, boolean switchDirection) {
		super(id, raster, x, y);
		this.width = width;
		this.height = height;
		this.image = image;
		this.duration = duration;
		this.switchDirection = switchDirection;
		alpha = 255;
		r = g = b = 255;	// default white
		startTime = System.currentTimeMillis();
	}

	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			c.pushMatrix();
			c.tint(r,g,b,alpha);
			if(switchDirection){
				y = height;
				x = c.width - (((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+width)) + width;
				if(x <= 0-width){
					die();
				}
				c.translate(x,y);
				c.rotate((float)Math.PI);	// flip it
			} else {
				y = 0;
				x = (((System.currentTimeMillis() - startTime) / (float)duration) * c.width) - width;
				if(x >= c.width){
					die();
				}
				c.translate(x,y);
			}
			c.image(image, 0, 0, width, height);
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
