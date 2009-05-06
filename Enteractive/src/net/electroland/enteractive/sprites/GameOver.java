package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class GameOver extends Sprite {
	
	private long startTime;
	private int expandDuration;
	private int fadeDuration;
	private int diameter = 1;
	private int brightness = 255;
	private boolean fadeMode = false;

	public GameOver(int id, Raster raster, float x, float y, SoundManager sm) {
		super(id, raster, x, y, sm);
		startTime = System.currentTimeMillis();
		expandDuration = 1000;
		fadeDuration = 2000;
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			
			if(!fadeMode){
				diameter = (int)(((System.currentTimeMillis() - startTime) / (float)expandDuration) * (c.width+(c.width/4)));
				if(System.currentTimeMillis() - startTime > expandDuration){
					fadeMode = true;
					startTime = System.currentTimeMillis();
				}
			} else {
				brightness = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)fadeDuration) * 255);
				if(System.currentTimeMillis() - startTime > fadeDuration){
					die();
				}
			}
			
			c.pushMatrix();
			c.ellipseMode(PConstants.CENTER);
			c.fill(brightness,0,0,255);
			c.ellipse(c.width/2, c.height/2, diameter, diameter);
			c.popMatrix();
		}
	}

}
