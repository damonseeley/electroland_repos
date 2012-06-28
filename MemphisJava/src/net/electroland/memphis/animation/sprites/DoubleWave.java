package net.electroland.memphis.animation.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;

public class DoubleWave extends Sprite{
	
	private PImage image, image2;
	private int duration;
	private int fadeDuration;
	private long startTime;
	private float alpha;

	public DoubleWave(int id, Raster raster, float x, float y, PImage image, PImage image2, float width, float height, int duration, int fadeDuration) {
		super(id, raster, x, y);
		this.image = image;
		this.image2 = image2;
		this.width = width;
		this.height = height;
		this.duration = duration;
		this.fadeDuration = fadeDuration;
		startTime = System.currentTimeMillis();
	}

	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			alpha = 255 - (((System.currentTimeMillis() - startTime) / (float)fadeDuration) * 255);
			//System.out.println("double wave alpha: "+ alpha);
			if(alpha <= 0){
				die();
				//System.out.println("double wave DEAD");
			}
			
			// left wave
			c.pushMatrix();
			c.tint(255,0,0,alpha);
			//float ypos = height;
			float xpos = (((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+width));
			//System.out.println("left wave - xpos: "+ xpos +" ypos: "+ ypos);
			//c.translate(x,ypos);
			//c.rotate((float)Math.PI);	// flip it
			c.translate(x,y);
			c.image(image2, 0-xpos, 0, xpos, height);
			c.popMatrix();
			

			// right wave
			c.pushMatrix();
			c.tint(255,0,0,alpha);
			//float xpos2 = (((System.currentTimeMillis() - startTime) / (float)duration) * c.width);
			c.translate(x,y);
			c.image(image, 0, 0, xpos, height);
			c.tint(255,255,255,255);	// set back to opaque, since processing has a bug with tint
			c.popMatrix();
		}
	}

}
