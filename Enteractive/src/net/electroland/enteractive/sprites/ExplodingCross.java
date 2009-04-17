package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class ExplodingCross extends Sprite {
	
	private int duration;
	private long startTime;
	private boolean expand, hold;
	private int length;

	public ExplodingCross(int id, Raster raster, int x, int y, SoundManager sm, int duration) {
		super(id, raster, x, y, sm);
		this.duration = duration;
		startTime = System.currentTimeMillis();
		expand = true;
		length = tileSize/2;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("explodingCross"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CORNER);
			c.fill(255,0,0);
			c.noStroke();
			
			if(expand){
				length = (int)(((System.currentTimeMillis() - startTime) / ((float)duration/3)) * c.width);
				if(length >= c.width){
					expand = false;
					hold = true;
				}
				c.rect(x*tileSize - tileSize/2, y*tileSize - tileSize/2, tileSize, length);			// expanding down
				c.rect(x*tileSize - tileSize/2, y*tileSize-length - tileSize/2, tileSize, length);	// moving up
				c.rect(x*tileSize - tileSize/2, y*tileSize - tileSize/2, length, tileSize);			// expanding right
				c.rect(x*tileSize-length - tileSize/2, y*tileSize - tileSize/2, length, tileSize);	// moving left
			} else if(hold){
				if(System.currentTimeMillis() - startTime > ((float)duration/3)){
					hold = false;
				}
			} else {
				length = c.width - ((int)(((System.currentTimeMillis() - startTime) / ((float)duration/3)) * c.width) - c.width);
				if(length <= 0){
					die();
				}
				c.rect(x*tileSize - tileSize/2, y*tileSize + (c.width-length), tileSize, length);				// moving down
				c.rect(x*tileSize - tileSize/2, y*tileSize-(tileSize/2)-c.width, tileSize, length);				// sliding up
				c.rect(x*tileSize - tileSize/2  + (c.width-length), y*tileSize - tileSize/2, length, tileSize);	// moving right
				c.rect(x*tileSize-c.width - tileSize/2, y*tileSize - tileSize/2, length, tileSize);				// sliding left
			}
			c.popMatrix();
		}
	}

}
