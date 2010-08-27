package net.electroland.memphis.animation.sprites;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.memphis.core.BridgeState;

public class Shooters extends Sprite implements SpriteListener {

	private ConcurrentHashMap<Integer,Sprite> sprites;
	private PImage image;
	private int duration;
	private long startTime;
	private long fadeStartTime;
	private long totalStartTime;
	private int totalDuration;
	private int shooterFrequency;
	private int shooterBrightness;
	private int spriteIndex = 0;
	private long fadeDuration;
	private int fadeIncrease;
	private BridgeState state;
	private int bay;
	private boolean fadeOutAndDie = false;
	private float occupiedThreshold = 0.5f;

	public Shooters(int id, Raster raster, float x, float y, PImage image, float width, float height, int duration, int shooterFrequency, int shooterBrightness, BridgeState state, int bay) {
		super(id, raster, x, y);
		this.width = width;
		this.height = height;
		this.image = image;
		this.duration = duration;
		this.shooterFrequency = shooterFrequency;
		this.shooterBrightness = shooterBrightness;
		this.state = state;
		this.bay = bay;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		totalDuration = 2000;
		fadeDuration = 1000;
		fadeIncrease = 100;	// millisecond increase per second
		startTime = System.currentTimeMillis();
		totalStartTime = System.currentTimeMillis();
		fadeStartTime = System.currentTimeMillis();
	}

	public void draw() {		
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)raster.getRaster();
			// calculate fade out time for each shooter
			//fadeDuration = (long)(((System.currentTimeMillis() - fadeStartTime) / 1000.0f) * fadeIncrease);
			
			if(System.currentTimeMillis() - totalStartTime < totalDuration){
				// see if it's time to create a new shooter
				if(System.currentTimeMillis() - startTime > shooterFrequency && !fadeOutAndDie){
					//System.out.println((System.currentTimeMillis() - fadeStartTime)/1000.0f + " fadeDuration: "+fadeDuration);
					float x = bay * (c.width/27);
					//float y = (float)Math.floor((Math.random() * 4)) * c.height/4;
					boolean flip = false;
					if(Math.random() > 0.5){
						flip = true;
					}
					Shooter shooter = new Shooter(spriteIndex, raster, image, 0, 0, width, height, duration, flip);
					shooter.setFadeDuration((int)fadeDuration);
					//shooter.setFadeDuration(500);
					/*
					if(flip){	// blue hues
						shooter.setColor(0.0f, (float)Math.random() * shooterBrightness, shooterBrightness);
					} else {	// green hues
						shooter.setColor(0.0f, (float)Math.random() * shooterBrightness, shooterBrightness);
						//shooter.setColor(0.0f, shooterBrightness, (float)Math.random() * shooterBrightness);
					}
					*/
					float saturation = (float)Math.random() * shooterBrightness;
					shooter.setColor(shooterBrightness, saturation, saturation);
					shooter.addListener(this);
					sprites.put(spriteIndex, shooter);
					spriteIndex++;
					startTime = System.currentTimeMillis();
				}
			}
			
			c.pushMatrix();
			// draw shooters
			c.translate(x, 0);
			Iterator<Sprite> iter = sprites.values().iterator();
			while(iter.hasNext()){
				Sprite sprite = (Sprite)iter.next();
				sprite.draw();
			}
			c.popMatrix();
		}
		/*
		if(!state.isOccupied(bay)){
			fadeOutAndDie = true;
		}
		*/
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
		if(sprites.size() == 0){
			die();
		}
	}

}
