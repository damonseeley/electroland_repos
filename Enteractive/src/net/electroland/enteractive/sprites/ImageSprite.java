package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class ImageSprite extends Sprite {
	
	private PImage image;
	private int imageWidth, imageHeight;

	public ImageSprite(Raster raster, int x, int y, PImage image, float imageWidth, float imageHeight) {
		super(raster, x, y);
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.imageWidth = (int)(imageWidth*c.height);		// always compare ratio to canvas HEIGHT
			this.imageHeight = (int)(imageHeight*c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.image(image, x-(imageWidth/2), y-(imageHeight/2), imageWidth, imageHeight);
			c.popMatrix();
		}
	}

}
