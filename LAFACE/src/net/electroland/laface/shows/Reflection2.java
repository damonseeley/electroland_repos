package net.electroland.laface.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.laface.core.LAFACEMain;
//import net.electroland.laface.core.Sprite;
//import net.electroland.laface.sprites.Bars;
import net.electroland.laface.tracking.Mover;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Reflection2 implements Animation {
	
	private Raster r;
	//private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	//private int spriteIndex = 0;
	//private Bars bars;
	private LAFACEMain main;
	
	public Reflection2(LAFACEMain main, Raster r){
		this.main = main;
		this.r = r;
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
					c.rect(m.getX()*c.width, 0, 50, c.height);
				}
			}
			
			//Iterator<Sprite> iter = sprites.values().iterator();
			//while(iter.hasNext()){
			//	Sprite sprite = (Sprite)iter.next();
			//	sprite.draw(r);
			//}
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
