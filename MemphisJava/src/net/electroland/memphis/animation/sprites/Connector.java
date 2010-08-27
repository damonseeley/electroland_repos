package net.electroland.memphis.animation.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;

public class Connector extends Sprite {
	
	private PImage image;
	private int duration, fadeDuration, timeout;
	private long startTime, fadeStartTime, timeoutStartTime;
	private boolean hold, fadeOut;
	private float r, g, b, alpha;
	private float width, height;
	private float fullLength;
	private int posA, posB;	// bay number
	private int locA, locB;	// horizontal pixel number

	public Connector(int id, Raster raster, float x, float y, int posA, int posB, int duration, int timeout, int fadeDuration) {
		super(id, raster, x, y);
		this.posA = posA;
		this.posB = posB;
		this.duration = duration;
		this.fadeDuration = fadeDuration;
		this.timeout = timeout;
		alpha = 255;
		r = 255;
		g = 255;
		b = 0;
		fadeDuration = 250;
		fadeOut = false;
		hold = false;
		height = 6;
		PGraphics c = (PGraphics)raster.getRaster();
		locA = (c.width / 27) * posA;
		locB = (c.width / 27) * posB;
		if(posA > posB){
			fullLength = (c.width / 27) * (posA - posB);
		} else {
			fullLength = (c.width / 27) * (posB - posA);
		}
		startTime = System.currentTimeMillis();
	}

	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			c.noStroke();
			
			// animation logic
			if(System.currentTimeMillis() - startTime < duration){	// if still stretching bar...
				width = ((System.currentTimeMillis() - startTime) / (float)duration) * fullLength;
			} else {			// if done stretching...
				if(!hold){		// turn the hold on
					hold = true;
					timeoutStartTime = System.currentTimeMillis();
				} else {		// if holding...
					if(System.currentTimeMillis() - timeoutStartTime > timeout){	// timeout has been reached
						hold = false;
						fadeOut = true;
						fadeStartTime = System.currentTimeMillis();
					}
				}
			}
			
			if(fadeOut){	// if fading out...
				alpha = 255 - (((System.currentTimeMillis() - fadeStartTime) / (float)fadeDuration) * 255);
				if(alpha <= 0){
					die();
				}
			}
			
			// animation temporal qualities
			c.pushMatrix();
			c.fill(r,g,b,alpha);
			if(posA > posB){
				// slide from posA to posB
				//System.out.println("locA: "+ locA +"locB:"+ locB);
				c.rect(locA-width, y, width, height);
			} else {
				// slide from posA to posB
				//System.out.println("posA: "+ posA +"posB:"+ posB);
				c.rect(locA, y, width, height);
			}
			//c.tint(255,255,255,255);	// set back to opaque, since processing has a bug with tint
			c.popMatrix();
		}
	}
	
	public void setColor(float r, float g, float b){
		this.r = r;
		this.g = g;
		this.b = b;
	}
	
	public void fadeOutAndDie(){
		hold = false;
		fadeOut = true;
		fadeStartTime = System.currentTimeMillis();
	}

}
