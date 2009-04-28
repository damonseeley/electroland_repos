package net.electroland.laface.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.laface.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Buoy extends Sprite {
	
	private long startTime;
	private int minDuration, maxDuration, duration;
	private PImage texture;
	private int alpha;
	private int ytarget;

	public Buoy(int id, Raster raster, float x, float y, PImage texture) {
		super(id, raster, x, y);
		this.texture = texture;
		alpha = 255;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			ytarget = (int)(Math.random()*c.height);
		}
		duration = (int)(Math.random()*(maxDuration - minDuration)) + minDuration;
		startTime = System.currentTimeMillis();
	}

	@Override
	public void draw(Raster r) {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.tint(255,255,255,alpha);
			c.image(texture, x, y, width, height);
		}
	}

}
