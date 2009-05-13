package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class BombDrop extends Sprite {
	
	private long startTime;
	private int duration, delay;
	private int alpha;
	private int gridWidth, gridHeight;
	private boolean wait = true;
	private Person person;

	public BombDrop(int id, Raster raster, float x, float y, SoundManager sm, Person person, int delay, int duration) {
		super(id, raster, x, y, sm);
		this.person = person;
		this.delay = delay;
		this.duration = duration;
		startTime = System.currentTimeMillis();
		alpha = 255;
		gridWidth = 18;
		gridHeight = 11;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.rectMode(PConstants.CENTER);
			sm.createMonoSound(sm.soundProps.getProperty("noise"), x, y, c.width, c.height);
		}
	}

	@Override
	public void draw() {
		if(wait && System.currentTimeMillis() - startTime < delay){
			/*
			// slowly fill graphic with black
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				c.pushMatrix();
				alpha = (int)(((System.currentTimeMillis() - startTime) / (float)delay) * 255);
				c.fill(0,0,0,alpha);
				c.rect(c.width/2,c.height/2,c.width,c.height);
				c.popMatrix();
			}
			*/
			
			// slowly enclose upon the pad activating the bomb drop using large rectangles to fill the space
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				c.pushMatrix();
				c.fill(150,0,0,255);
				// left side
				int leftwidth = (int)(((System.currentTimeMillis() - startTime) / (float)delay) * (person.getX()*tileSize));
				c.rect(leftwidth/2, c.height/2, leftwidth, c.height);
				// right side
				int rightwidth = (int)(((System.currentTimeMillis() - startTime) / (float)delay) * (c.width-(person.getX()*tileSize)));
				c.rect(c.width-rightwidth + (rightwidth/2), c.height/2, rightwidth, c.height);
				// top
				int topheight = (int)(((System.currentTimeMillis() - startTime) / (float)delay) * (person.getY()*tileSize));
				c.rect(c.width/2, topheight/2, c.width, topheight);
				// bottom 
				int bottomheight = (int)(((System.currentTimeMillis() - startTime) / (float)delay) * (c.height - (person.getY()*tileSize)));
				c.rect(c.width/2, c.height - (bottomheight/2), c.width, bottomheight);
				
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
