package net.electroland.laface.shows;

import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.core.Sprite;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Floaters implements Animation {
	
	private Raster r;
	private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	private LAFACEMain main;
	private PImage texture;
	
	public Floaters(LAFACEMain main, Raster r, PImage texture){
		this.main = main;
		this.r = r;
		this.texture = texture;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);	
	}

	public Raster getFrame() {
		return r;
	}
	
	public void cleanUp() {
	}

	public boolean isDone() {
		return false;
	}

}
