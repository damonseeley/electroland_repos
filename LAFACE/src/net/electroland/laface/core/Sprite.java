package net.electroland.laface.core;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Raster;

/**
 * This is used for sub-show animation objects.
 * @author asiegel
 */

public abstract class Sprite {
	
	private List <SpriteListener> listeners;
	protected Raster raster;		// Raster passed into sprite
	protected Object canvas;		// PGraphics or Graphics object used to draw on
	protected float x, y, width, height;
	protected int tileSize;		// size of tile relative to raster pixel dimensions
	protected int id;
	
	public Sprite(int id, Raster raster, float x, float y){
		this.id = id;
		this.raster = raster;
		this.x = x;
		this.y = y;
		if(raster.isProcessing()){
			canvas = (PGraphics)raster.getRaster();
		}
		this.tileSize = (int)(((PGraphics)canvas).height/11.0);
		listeners = new ArrayList<SpriteListener>();
	}
	
	abstract public void draw(Raster r);	// show calls this every frame
	
	final public void addListener(SpriteListener listener){
		listeners.add(listener);
	}

	final public void removeListener(SpriteListener listener){
		listeners.remove(listener);
	}
	
	final public void die(){
		// tell any listeners that we are done.
		Iterator<SpriteListener> i = listeners.iterator();
		while (i.hasNext()){
			i.next().spriteComplete(this);
		}
	}
	
	public int getID(){
		return id;
	}
	
	public float getX(){
		return x;
	}
	
	public float getY(){
		return y;
	}
	
	final public void moveTo(int x, int y){
		this.x = x;
		this.y = y;
	}
	
	final public void tweenTo(float xpos, float ypos, int durationMs){
		// TODO set new X/Y vectors and tween to the position over the duration
	}

	/* VISUAL EFFECTS */
	
	final public void throb(int min, int max, int speedMs){
		// TODO throb the sprite brightness between min and max values
	}
	

}
