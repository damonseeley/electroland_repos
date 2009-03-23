package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

//	###
//	# #
//	###

public class Box extends Sprite{
	
	public Box(int id, Raster raster, int x, int y, int width, int height){
		super(id, raster, x, y);
		this.width = tileSize*width;
		this.height = tileSize*height;					// using tile size to scale sprite size
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			//c.beginDraw();							// may not be necessary since show is required to do this
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);				// centered at sprite's X/Y position
			c.fill(255,0,0);
			c.rect(x-tileSize, y, height, tileSize);	// left side
			c.rect(x+tileSize, y, height, tileSize);	// right side
			c.rect(x, y+tileSize, tileSize, tileSize);	// top square
			c.rect(x, y-tileSize, tileSize, tileSize);	// bottom square
			c.popMatrix();
			//c.endDraw();								// may not be necessary since show is required to do this
		}
	}

}
