package net.electroland.enteractive.sprites;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.lighting.detector.animation.Raster;

public class Sparkler extends Sprite implements SpriteListener{
	
	private Person person;
	private PImage image;
	private ConcurrentHashMap<Integer,Spark> sparks;
	private int timeOut;
	private long startTime;
	private boolean timeToDie;
	private int sparkIndex = 0;	// used as ID # for spark
	private float sparkOdds = 0.9f;
	private int maxSparks = 5;

	public Sparkler(int id, Raster raster, float x, float y, SoundManager sm, Person person, PImage image) {
		super(id, raster, x, y, sm);
		this.person = person;
		this.image = image;
		sparks = new ConcurrentHashMap<Integer,Spark>();
		timeOut = 10000;
		timeToDie = false;
		startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(sparks.size() < maxSparks && !timeToDie){
			if(Math.random() > sparkOdds){
				Spark spark = new Spark(sparkIndex, raster, x, y, sm, image);
				spark.addListener(this);
				sparks.put(sparkIndex, spark);
				sparkIndex++;
			}
		} else if (sparks.size() == 0 && timeToDie){
			die();
		}
		
		if(((person != null && person.isDead()) || System.currentTimeMillis() - startTime > timeOut) && !timeToDie){
			startTime = System.currentTimeMillis();
			timeToDie = true;
		}
	
		Iterator<Spark> iter = sparks.values().iterator();
		while(iter.hasNext()){
			Spark sprite = iter.next();
			sprite.draw();
		}
	}
	
	@Override
	public void spriteComplete(Sprite sprite) {
		sparks.remove(sprite.getID());
	}
	
	
	
	
	
	
	public class Spark extends Sprite{
		
		private int alpha = 255;
		private boolean fadeOut = false;
		private int imageWidth, imageHeight;
		private long sparkStartTime;
		private float xdest, ydest, xstart, ystart;
		private float sparkLife = 300;	// milliseconds
		private PImage sparkImage;

		public Spark(int id, Raster raster, float x, float y, SoundManager sm, PImage sparkImage) {
			super(id, raster, x, y, sm);
			this.sparkImage = sparkImage;
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				imageWidth = imageHeight = c.height/4;
				xdest = (float)(Math.random()*c.width);
				ydest = (float)(Math.random()*c.height);
				xstart = x;
				ystart = y;
				sparkStartTime = System.currentTimeMillis();
				sm.createMonoSound(sm.soundProps.getProperty("test1"), x, y, c.width, c.height);
			}
		}

		@Override
		public void draw() {
			if(raster.isProcessing()){
				PGraphics c = (PGraphics)canvas;
				c.pushMatrix();
				if(alpha < 255){
					c.tint(255,255,255,alpha);
				}
				c.image(sparkImage, x-(imageWidth/2), y-(imageHeight/2), imageWidth, imageHeight);
				c.popMatrix();
			}
			
			x = (int)(((System.currentTimeMillis() - sparkStartTime) / (float)sparkLife) * (xdest-xstart));
			y = (int)(((System.currentTimeMillis() - sparkStartTime) / (float)sparkLife) * (ydest-ystart));
			
			if(fadeOut){
				//alpha = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255); // linear fade out
				alpha = (int)(255 - Math.sin((Math.PI/2) * ((System.currentTimeMillis() - sparkStartTime) / (float)sparkLife))*255);	// dampened fade out
				if(alpha <= 0){
					die();
				}
			}
		}
		
	}

}
