package net.electroland.memphis.animation.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;

public class Cloud extends Sprite {
	
	private PImage image;
	private int duration;
	private long startTime;
	private float r, g, b, alpha;

	public Cloud(int id, Raster raster, float x, float y, PImage image, int duration) {
		super(id, raster, x, y);
		//System.out.println(x);
		this.image = image;
		this.duration = duration;
		width = image.width;
		height = image.height;
		alpha = 255;
		r = g = b = 255;	// default white
		startTime = System.currentTimeMillis();
	}

	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			c.pushMatrix();
			c.tint(r,g,b,alpha);
			x = 0 - (((System.currentTimeMillis() - startTime) / (float)duration) * (width/2));
			if(x <= 0 - (width/2)){
				//die();
				x = 0;
				startTime = System.currentTimeMillis();
				//System.out.println("repeat");
			}
			//System.out.println(x);
			c.image(image, x, 0, width, height);
			c.tint(255,255,255,255);	// set back to opaque, since processing has a bug with tint
			c.popMatrix();
		}
	}
	
	public void setColor(float r, float g, float b, float alpha){
		this.r = r;
		this.g = g;
		this.b = b;
		this.alpha = alpha;
	}

}
