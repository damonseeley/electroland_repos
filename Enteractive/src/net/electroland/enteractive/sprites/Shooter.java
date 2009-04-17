package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Shooter extends Sprite {

	private PImage image;
	private int duration;
	private long startTime;
	private boolean switchDirection;
	private int sweepLength;
	
	public Shooter(int id, Raster raster, float x, float y, SoundManager sm, PImage image, int duration, boolean switchDirection) {
		super(id, raster, x, y, sm);
		this.image = image;
		this.duration = duration;
		this.switchDirection = switchDirection;
		sweepLength = 150;
		startTime = System.currentTimeMillis();
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("shooter"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.tint(255,255,255,255);
			if(switchDirection){
				x = c.width - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
				if(x <= 0-sweepLength){
					die();
				}
				c.translate(x, y);
				c.rotate((float)Math.PI);	// flip it
				c.image(image, 0-sweepLength,  0-(tileSize/2), sweepLength, tileSize);
			} else {
				x = (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
				if(x >= c.width+sweepLength){
					die();
				}
				c.image(image, x-sweepLength, y-(tileSize/2), sweepLength, tileSize);
			}
			c.popMatrix();
		}
	}

}
