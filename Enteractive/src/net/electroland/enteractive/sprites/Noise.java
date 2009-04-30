package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Noise extends Sprite {
	
	private long startTime;
	private int duration, delay;
	private int alpha;
	private int gridWidth, gridHeight;
	private boolean wait = true;

	public Noise(int id, Raster raster, float x, float y, SoundManager sm, int delay, int duration) {
		super(id, raster, x, y, sm);
		this.delay = delay;
		this.duration = duration;
		startTime = System.currentTimeMillis();
		alpha = 255;
		gridWidth = 18;
		gridHeight = 11;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("noise"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(wait && System.currentTimeMillis() - startTime < delay){
			// slowly fill graphic with black
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				c.pushMatrix();
				c.fill(0,0,0,(int)(((System.currentTimeMillis() - startTime) / (float)delay) * 255));
				c.rect(0,0,c.width,c.height);
				c.popMatrix();
			}
		} else if (wait && System.currentTimeMillis() - startTime >= delay){
			startTime = System.currentTimeMillis();
			wait = false;
		} else {
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				c.pushMatrix();
				for(int y=0; y<gridHeight; y++){
					for(int x=0; x<gridWidth; x++){
						c.fill((int)(Math.random()*255), 0, 0, alpha);
						c.rect(x*tileSize, y*tileSize, tileSize, tileSize);
					}
				}
				c.popMatrix();
			}
			alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255);
			if(alpha <= 0){
				die();
			}
		}
	}

}
