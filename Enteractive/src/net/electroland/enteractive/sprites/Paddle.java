package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Paddle extends Sprite {

	public Paddle(int id, Raster raster, float x, float y, SoundManager sm) {
		super(id, raster, x, y, sm);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			c.popMatrix();
		}
	}

}
