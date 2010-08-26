package net.electroland.memphis.animation.sprites;

import net.electroland.lighting.detector.animation.Raster;
import processing.core.PGraphics;
import processing.core.PImage;

public class Shooter extends Sprite {

	private PImage image;
	private int duration, fadeDuration;
	private long startTime, fadeStartTime;
	private boolean switchDirection, fadeOut;
	private float r, g, b, alpha;
	
	public Shooter(int id, Raster raster, PImage image, float x, float y, float width, float height, int duration, boolean switchDirection){
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
	
	public void draw(){
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			c.pushMatrix();
			c.tint(r,g,b,alpha);
			//System.out.println(id +" x: "+ x +" y: "+ y +" width: "+ width +" height: "+ height);
			if(switchDirection){
				x = 0 - (((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+width));
				if(x <= 0-(c.width+width)){
					die();
				}
				c.translate(x, y);
				c.rotate((float)Math.PI);	// flip it
				c.image(image, 0-width,  0-(height/2), width, height);
				//c.image(image, 0-sweepLength,  0, sweepLength, height);
			} else {
				x = (((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+width));
				if(x >= c.width+width){
					die();
				}
				c.image(image, x-width, y-(height/2), width, height);
				//c.image(image, x-sweepLength, y, sweepLength, height);
			}
			if(fadeOut){
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
	
	public void setFadeDuration(int fadeDuration){
		this.fadeDuration = fadeDuration;	
	}
	
}
