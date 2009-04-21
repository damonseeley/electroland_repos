package net.electroland.enteractive.sprites;

import java.util.Iterator;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.lighting.detector.animation.Raster;

public class Radar extends Sprite{
	
	private LilyPad show;
	private PImage texture;
	private int radius, rotSpeed, rotation;
	private long startTime;

	public Radar(int id, Raster raster, float x, float y, SoundManager sm, LilyPad show, PImage texture, int radius, int rotSpeed) {
		super(id, raster, x, y, sm);
		this.show = show;
		this.texture = texture;
		this.radius = radius;
		this.rotSpeed = rotSpeed;
		this.startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.translate(x, y);
			c.rotate((float)(Math.PI/180)*rotation);
			c.image(texture, 0-radius, 0-radius, radius*2, radius*2);
			c.popMatrix();
		}
		
		// TODO each rotation, get unit vector of radar sweep
		
		// check for collision against an indicator sprite
		Iterator<Single> singleiter = show.billiejean.values().iterator();
		while(singleiter.hasNext()){
			Sprite sprite = (Sprite)singleiter.next();
			// TODO compare indicator position to unit vector of sweep
			// TODO play sound if they match up
		}
	}

}
