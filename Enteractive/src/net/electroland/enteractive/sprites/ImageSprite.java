package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class ImageSprite extends Sprite {
	
	private PImage image;
	private int imageWidth, imageHeight;
	private boolean expand, fadeOut;
	private int duration;
	private int startSize, endSize;
	private int alpha = 255;
	private long startTime;

	public ImageSprite(int id, Raster raster, float x, float y, SoundManager sm, PImage image, float imageWidth, float imageHeight) {
		super(id, raster, x, y, sm);
		this.image = image;			// percent of raster height as a float
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.imageWidth = (int)(imageWidth*c.height);		// always compare ratio to canvas HEIGHT
			this.imageHeight = (int)(imageHeight*c.height);
			this.startSize = this.imageWidth;
			this.endSize = c.height*2;
		}
		expand = true;
		duration = 750;	// milliseconds
		fadeOut = true;
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
			alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255);
			if(alpha <= 0){
				die();
			}
		}		
	}

}
