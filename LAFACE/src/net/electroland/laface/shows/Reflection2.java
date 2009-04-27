package net.electroland.laface.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.tracking.Target;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Reflection2 implements Animation {
	
	private Raster r;
	//private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	//private int spriteIndex = 0;
	//private Bars bars;
	private LAFACEMain main;
	private PImage texture;
	private int xscale;	// canvas width + margins where camera image is not in front of lighting grid
	private int xoffset;	// compensating for margin
	private boolean blobSizeMode = true;	// TODO toggle this to make sprite size dynamic
	
	public Reflection2(LAFACEMain main, Raster r, PImage texture){
		this.main = main;
		this.r = r;
		this.texture = texture;
		//sprites = new ConcurrentHashMap<Integer,Sprite>();
		//bars = new Bars(spriteIndex, r, 0, 0);
		//sprites.put(spriteIndex, bars);
		//spriteIndex++;
	}
	
	public void initialize() {
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
					int width = 50;
					if(blobSizeMode){
						width = (int)(t.getWidth()*xscale);
					}
					//c.image(texture, ((int)((t.getX()*xscale) - (width/2))) - xoffset, 0, width, c.height);	// for testing via gui
					c.image(texture, (xscale - ((int)((t.getX()*xscale) - (width/2)))) - xoffset , 0, width, c.height);		// must mirror output for lights
				}
				//System.out.println("targets: "+ targets.size() + " alive: "+alive);
			//}
			
//			Iterator<Sprite> iter = sprites.values().iterator();
//			while(iter.hasNext()){
//				Sprite sprite = (Sprite)iter.next();
//				sprite.draw(r);
//			}
			c.endDraw();
		}
		return r;
	}
	
	public void cleanUp() {
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

}
