package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

//	 #
//	###
//	 #

public class Cross extends Sprite{
	
	private Person person;
	
	public Cross(int id, Raster raster, Person person, int x, int y, SoundManager sm, int width, int height){
		super(id, raster, x, y, sm);
		this.person = person;
		this.width = tileSize*width;
		this.height = tileSize*height;				// using tile size to scale sprite size
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("cross"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);			// centered at sprite's X/Y position
			c.fill(255,0,0);
			c.noStroke();
			c.rect(x, y, width, tileSize);			// horizontal rectangle
			c.rect(x, y, tileSize, height);			// vertical rectangle
			c.popMatrix();
			if(person != null && person.isDead()){
				die();
			}
		}
	}

}
