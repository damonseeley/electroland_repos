package net.electroland.memphis.animation.sprites;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import net.electroland.lighting.detector.animation.Raster;

public abstract class Sprite {
	
	protected int id;
	protected float x, y, width, height;
	protected Raster raster;		// Raster passed into sprite
	private List <SpriteListener> listeners;
	
	public Sprite(int id, Raster raster, float x, float y){
		this.id = id;
		this.raster = raster;
		this.x = x;
		this.y = y;
		listeners = new ArrayList<SpriteListener>();
	}
	
	abstract public void draw();	// show calls this every frame
	
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

}
