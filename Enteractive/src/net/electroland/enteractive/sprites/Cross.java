package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

//	 #
//	###
//	 #

public class Cross extends Sprite{
	
	public Cross(Raster raster, int x, int y, int width, int height){
		super(raster, x, y);
		this.width = tileSize*width;
		this.height = tileSize*height;				// using tile size to scale sprite size
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			//c.beginDraw();						// may not be necessary since show is required to do this
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			c.fill(255,0,0);
			c.rect(x, y, width, tileSize);			// horizontal rectangle
			c.rect(x, y, tileSize, height);			// vertical rectangle
			c.popMatrix();
			//c.endDraw();							// may not be necessary since show is required to do this
		}
	}

}
