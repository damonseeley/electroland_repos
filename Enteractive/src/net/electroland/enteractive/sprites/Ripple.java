package net.electroland.enteractive.sprites;

import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;
import processing.core.PGraphics;
import processing.core.PImage;

public class Ripple extends Sprite{

	private PImage image;
	private int imageWidth, imageHeight;
	private boolean expand, fadeOut;
	private int duration;
	private int startSize, endSize;
	private int alpha = 255;
	private long startTime;
	
	public Ripple(int id, Raster raster, float x, float y, SoundManager sm, PImage image){
		super(id, raster, x, y, sm);
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.startSize = this.imageWidth = this.imageHeight = (int)(c.height/11);
			this.endSize = c.height*2;
			sm.createMonoSound(sm.soundProps.getProperty("ripple"), x, y, c.width, c.height);
		}
		expand = true;
		fadeOut = true;
		duration = 750;	// milliseconds
		startTime = System.currentTimeMillis();
	}
	
	public Ripple(int id, Raster raster, float x, float y, SoundManager sm, PImage image, float endSize, String soundCue){
		super(id, raster, x, y, sm);
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.startSize = this.imageWidth = this.imageHeight = (int)(c.height/11);
			this.endSize = (int)(c.height*endSize);
			sm.createMonoSound(sm.soundProps.getProperty(soundCue), x, y, c.width, c.height);
		}
		expand = true;
		fadeOut = true;
		duration = 750;	// milliseconds
		startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			if(alpha < 255){
				c.tint(255,255,255,alpha);
			}
			c.image(image, (x*tileSize)-(imageWidth/2), (y*tileSize)-(imageHeight/2), imageWidth, imageHeight);
			c.popMatrix();
		}
		
		if(expand){
			imageWidth = imageHeight = startSize + (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (endSize-startSize));
		}
		if(fadeOut){
			//alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255); // linear fade out
			alpha = (int)(255 - Math.sin((Math.PI/2) * ((System.currentTimeMillis() - startTime) / (float)duration))*255);	// dampened fade out
			if(alpha <= 0){
				die();
			}
		}
	}
	
}
