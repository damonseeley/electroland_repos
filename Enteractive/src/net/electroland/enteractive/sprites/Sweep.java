package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Sweep extends Sprite {
	
	private PImage image;
	private int duration;
	private long startTime;
	private boolean switchDirection;
	private int sweepLength;

	public Sweep(int id, Raster raster, float x, float y, SoundManager sm, PImage image, int duration, boolean switchDirection) {
		super(id, raster, x, y, sm);
		this.image = image;
		this.duration = duration;
		this.switchDirection = switchDirection;
		sweepLength = 50;
		startTime = System.currentTimeMillis();
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("test1"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			//c.tint(255,255,255,255);
			if(switchDirection){
				x = c.width - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
				if(x <= 0-sweepLength){
					die();
				}
				c.translate(x, 0);
				c.rotate((float)Math.PI);	// flip it
				c.image(image, 0-sweepLength, 0-c.height, sweepLength, c.height);
			} else {
				x = (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
				if(x >= c.width+sweepLength){
					die();
				}
				c.image(image, x-sweepLength, 0, sweepLength, c.height);
			}
			c.popMatrix();
		}
	}

}
