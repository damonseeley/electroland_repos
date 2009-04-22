package net.electroland.laface.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.tracking.Mover;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Reflection2 implements Animation {
	
	private Raster r;
	//private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	//private int spriteIndex = 0;
	//private Bars bars;
	private LAFACEMain main;
	private PImage texture;
	private boolean blobSizeMode = false;	// TODO toggle this to make sprite size dynamic
	
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
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);	
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			
			synchronized(main.tracker){	// TODO is this safe?
				ConcurrentHashMap<Integer,Mover> movers = main.tracker.getMovers();
				Iterator<Mover> iter = movers.values().iterator();
				while(iter.hasNext()){
					Mover m = iter.next();
					// TODO width based on blob size
					//System.out.println(m.getX()*c.width);
					//c.rect(m.getX()*c.width, 0, 50, c.height);
					int width = 50;
					if(blobSizeMode){
						width = (int)m.getWidth();
					}
					if(m.getXvec() < 0){
						c.image(texture, (int)((m.getX()*c.width) - (width/2)), 0, width, c.height);
					} else {
						c.rotate((float) Math.PI);
						c.image(texture, 0 - (int)((m.getX()*c.width) - (width/2)), 0-c.height, width, c.height);
						c.rotate((float) Math.PI);
					}
				}
			}
			
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

}
