package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class ImageSprite extends Sprite {
	
	private PImage image;
	private int imageWidth, imageHeight;
	private boolean expand, fadeOut;
	private int expansionSpeed, fadeSpeed;
	private int alpha = 255;

	public ImageSprite(int id, Raster raster, float x, float y, PImage image, float imageWidth, float imageHeight) {
		super(id, raster, x, y);
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.imageWidth = (int)(imageWidth*c.height);		// always compare ratio to canvas HEIGHT
			this.imageHeight = (int)(imageHeight*c.height);
		}
		expand = true;
		expansionSpeed = 4;	// pixel per frame for now
		fadeOut = true;
		fadeSpeed = 5;		// color units per frame for now
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
			imageWidth += expansionSpeed;
			imageHeight += expansionSpeed;
		}
		if(fadeOut){
			alpha -= fadeSpeed;
			if(alpha <= 0){
				die();
			}
		}
	}

}
