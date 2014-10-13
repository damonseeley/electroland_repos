package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Single extends Sprite {
	
	private Person person;
	private int minVal, maxVal, brightness;

	public Single(int id, Raster raster, Person person, float x, float y, SoundManager sm) {
		super(id, raster, x, y, sm);
		this.person = person;
		minVal = 10;
		maxVal = 48;
		//maxVal = 200;
		brightness = maxVal;
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			//c.fill(150,0,0);						// dimmed
			//c.fill((float)(Math.random()*(maxVal-minVal))+minVal,0,0);
			if(brightness == minVal){
				brightness = maxVal;
			} else {
				brightness = minVal;
			}
			c.fill(brightness, 0, 0);
			c.noStroke();
			c.rect(x, y, tileSize, tileSize);		// single tile
			c.popMatrix();
			if(person != null && person.isDead()){
				die();
			}
		}
	}

}
