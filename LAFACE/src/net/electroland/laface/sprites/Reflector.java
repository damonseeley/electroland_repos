package net.electroland.laface.sprites;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.tracking.Target;
import net.electroland.lighting.detector.animation.Raster;

public class Reflector extends Sprite {
	
	private PImage texture;
	private Target target;

	public Reflector(int id, Raster raster, float x, float y, PImage texture, Target target) {
		super(id, raster, x, y);
		this.texture = texture;
		this.target = target;
	}

	@Override
	public void draw(Raster r) {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.image(texture, x, y, width, c.height);
		}
		if(target.isDead()){	// if fully dead, and removed from targets list...
			die();				// kill off the sprite // TODO make this into a fade-out effect
		}
	}
	
	public void setWidth(int width){
		this.width = width;
	}

}
