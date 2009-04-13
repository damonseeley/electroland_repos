package net.electroland.enteractive.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Pong implements Animation, SpriteListener {
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;
	private long startTime;
	private int duration = 10000;	// milliseconds
	
	public Pong(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
		startTime = System.currentTimeMillis();
	}

	public Raster getFrame() {
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			
			Iterator<Sprite> spriteiter = sprites.values().iterator();
			while(spriteiter.hasNext()){
				Sprite sprite = (Sprite)spriteiter.next();
				sprite.draw();
			}
			raster.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	public boolean isDone() {
		// TODO eventually based on game points
		return (System.currentTimeMillis() - startTime) >= duration;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}
	
	
	
	
	
	private class Player{
		private int points;		// points earned this game
		private int y1, y2;		// position of feet determines bar length
		private int x;
		
		private Player(boolean playerA){
			if(playerA){
				x = 1*tileSize;
			} else {
				x = 17*tileSize;
			}
		}
	}

}
