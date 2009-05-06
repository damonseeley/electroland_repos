package net.electroland.laface.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.sprites.Buoy;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Floaters implements Animation {
	
	private Raster r;
	private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	private LAFACEMain main;
	
	public Floaters(LAFACEMain main, Raster r, PImage texture){
		this.main = main;
		this.r = r;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.colorMode(PConstants.RGB, 255, 255, 255, 255);	
			for(int i=0; i<174; i++){
				sprites.put(i, new Buoy(i, r, (int)(i * (c.width/174.0f)), c.height, texture));
			}
		}
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			Iterator<Sprite> spriteiter = sprites.values().iterator();
			while(spriteiter.hasNext()){
				Sprite sprite = (Sprite)spriteiter.next();
				sprite.draw(r);
			}
			c.endDraw();
		}
		
		return r;
	}

	public boolean isDone() {
		return false;
	}

}
