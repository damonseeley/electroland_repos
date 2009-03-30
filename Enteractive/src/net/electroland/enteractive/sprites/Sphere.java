package net.electroland.enteractive.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Sphere extends Sprite{
	
	private Person person;
	private PImage image;
	private int imageWidth, imageHeight;
	private boolean fadeOut;
	private int duration;
	private int alpha = 200;
	private long startTime;

	public Sphere(int id, Raster raster, float x, float y, SoundManager sm, Person person, PImage image) {
		super(id, raster, x, y, sm);
		this.person = person;
		this.image = image;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			this.imageWidth = this.imageHeight = (int)(c.height/2);
			sm.createMonoSound(sm.soundProps.getProperty("test2"), x, y, c.width, c.height);
		}
		fadeOut = false;
		duration = 750;	// milliseconds
		startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			if(alpha < 255){
				c.tint(255,255,255,alpha);
			}
			c.image(image, x-(imageWidth/2), y-(imageHeight/2), imageWidth, imageHeight);
			c.popMatrix();
		}
		if(person.isDead() && !fadeOut){
			fadeOut = true;
			startTime = System.currentTimeMillis();
		}
		if(fadeOut){
			//alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255); 						// linear fade out
			alpha = (int)(255 - Math.sin((Math.PI/2) * ((System.currentTimeMillis() - startTime) / (float)duration))*255);	// dampened fade out
			if(alpha <= 0){
				die();
			}
		}
	}

}
