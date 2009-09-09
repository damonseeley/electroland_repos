package net.electroland.tracking.development.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.core.SpriteListener;
import net.electroland.laface.sprites.Reflector;
import net.electroland.laface.tracking.Target;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Reflection2 implements Animation, SpriteListener {
	
	private Raster r;
	private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	private LAFACEMain main;
	private PImage leftarrow, rightarrow;
	private int xscale;	// canvas width + margins where camera image is not in front of lighting grid
	private int xoffset;	// compensating for margin
	private boolean blobSizeMode = false;	// TODO toggle this to make sprite size dynamic
	
	public Reflection2(LAFACEMain main, Raster r, PImage leftarrow, PImage rightarrow){
		this.main = main;
		this.r = r;
		this.leftarrow = leftarrow;
		this.rightarrow = rightarrow;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		PGraphics c = (PGraphics)(r.getRaster());
		xoffset = 100;
		xscale = c.width + (xoffset*2);
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);	
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			
			//synchronized(main.tracker){	// TODO is this safe?
				ConcurrentHashMap<Integer,Target> targets = main.tracker.getTargets();
				int alive = 0;
				Iterator<Target> iter = targets.values().iterator();
				while(iter.hasNext()){
					Target t = iter.next();
					if(t.isTrackAlive()){
						alive++;
					}

					int width = c.height;
					if(blobSizeMode){
						width = (int)(t.getWidth()*xscale);
					}
					if(!sprites.containsKey(t.getID())){
						Reflector reflector = new Reflector(t.getID(), r, (xscale - ((int)((t.getX()*xscale) - (width/2)))) - xoffset, 0, leftarrow, rightarrow, t);
						reflector.setWidth(width);
						reflector.addListener(this);
						sprites.put(t.getID(), reflector);
					} else {
						Reflector reflector = (Reflector)sprites.get(t.getID());
						reflector.setWidth(width);
						reflector.moveTo((xscale - ((int)((t.getX()*xscale) - (width/2)))) - xoffset, 0);
					}
				}
				//System.out.println("sprites: " + sprites.size() + " targets: "+ targets.size() + " alive: "+alive);
			//}
			
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
	
	public void setXoffset(int xoffset){
		this.xoffset = xoffset;
	}
	
	public void setXscale(int xscale){
		this.xscale = xscale;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}

}
