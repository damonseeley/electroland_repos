package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Noise extends Sprite {
	
	private long startTime;
	private int alpha;
	private int gridWidth, gridHeight;

	public Noise(int id, Raster raster, float x, float y, SoundManager sm) {
		super(id, raster, x, y, sm);
		startTime = System.currentTimeMillis();
		alpha = 255;
		gridWidth = 18;
		gridHeight = 11;
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			for(int y=0; y<gridHeight; y++){
				for(int x=0; x<gridWidth; x++){
					c.fill((int)(Math.random()*255), 0, 0, alpha);
				}
			}
		}
	}

}
