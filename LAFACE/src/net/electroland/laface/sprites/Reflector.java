package net.electroland.laface.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.tracking.Target;
import net.electroland.lighting.detector.animation.Raster;

public class Reflector extends Sprite {
	
	private PImage leftarrow, rightarrow;
	private Target target;
	private int alpha;
	private long startTime;
	private int fadeDuration;
	private boolean fadeOut;

	public Reflector(int id, Raster raster, float x, float y, PImage leftarrow, PImage rightarrow, Target target) {
		super(id, raster, x, y);
		this.leftarrow = leftarrow;
		this.rightarrow = rightarrow;
		this.target = target;
		alpha = 255;
		fadeDuration = 1000;
		fadeOut = false;
	}

	@Override
	public void draw(Raster r) {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			if(!target.isTrackProvisional() && target.xpositions.size() > 1){
				c.tint(255,255,255,alpha);
				if(target.getXvec() < 0){
					c.image(rightarrow, x, y, width, c.height);
				} else {
					c.image(leftarrow, x, y, width, c.height);
				}
			}
		}
		if(target.isDead() && !fadeOut){	// if fully dead, and removed from targets list...
			startTime = System.currentTimeMillis();
			fadeOut = true;
		}
		if(fadeOut){
			if(System.currentTimeMillis() - startTime < fadeDuration){
				alpha = (int)(255 - (((System.currentTimeMillis() - startTime)/(float)fadeDuration) * 255));
			} else {
				die();
			}
		}
	}
	
	public void setWidth(int width){
		this.width = width;
	}

}
