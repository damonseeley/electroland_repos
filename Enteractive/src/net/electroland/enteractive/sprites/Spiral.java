package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Spiral extends Sprite {
	
	private Person person;
	private PImage image;
	private int imageWidth, imageHeight, maxDiameter, startDiameter;
	private boolean fadeOut, expand;
	private int timeOut, fadeSpeed, expandSpeed, rotSpeed, rotation;
	private int alpha = 255;
	private long startTime, rotTime;

	public Spiral(int id, Raster raster, float x, float y, SoundManager sm, Person person, PImage image) {
		super(id, raster, x, y, sm);
		this.person = person;
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			//this.imageWidth = this.imageHeight = (int)(c.width/2);
			//this.imageWidth = this.imageHeight = this.startDiameter = ((int)(c.width/11)) * 3;
			this.imageWidth = this.imageHeight = this.startDiameter = c.width;
			maxDiameter = ((int)(c.width/11)) * 32;
			sm.createMonoSound(sm.soundProps.getProperty("spiral"), x, y, c.width, c.height);
		}
		fadeOut = false;
		expand = false;
		timeOut = 3000;	// milliseconds
		fadeSpeed = 1000;
		expandSpeed = 3000;
		rotSpeed = 2000;
		rotTime = System.currentTimeMillis();
		startTime = System.currentTimeMillis();
		alpha = 255;
	}

	@Override
	public void draw() {
		if(System.currentTimeMillis() - rotTime > rotSpeed){
			rotTime = System.currentTimeMillis();
		}
		rotation = (int)(((System.currentTimeMillis() - rotTime) / (float)rotSpeed) * -360);
		
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.tint(255,255,255,alpha);
			c.translate(x, y);
			c.rotate((float)(rotation*(Math.PI/180)));
			c.image(image, 0-(imageWidth/2), 0-(imageHeight/2), imageWidth, imageHeight);
			c.popMatrix();
		}
		
		if(person.isDead() && !fadeOut){
			fadeOut = true;
			startTime = System.currentTimeMillis();
		}
		
		if(System.currentTimeMillis() - startTime > timeOut){
			fadeOut = true;
			startTime = System.currentTimeMillis();
		}
		
		if(expand){
			imageWidth = imageHeight = startDiameter + (int)(((System.currentTimeMillis() - startTime) / (float)expandSpeed) * maxDiameter-startDiameter);
		}
		
		if(fadeOut){
			//alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255); 						// linear fade out
			alpha = (int)(255 - Math.sin((Math.PI/2) * ((System.currentTimeMillis() - startTime) / (float)fadeSpeed))*255);	// dampened fade out
			if(alpha <= 0){
				die();
			}
		}
	}

}
