package net.electroland.enteractive.core;

import java.awt.Graphics;
import java.util.Iterator;
import java.util.List;

/**
 * This is used for sub-show animation objects.
 * @author asiegel
 */

public abstract class Sprite {
	
	private List <SpriteListener> listeners;
	protected float x, y, width, height;
	private float xvec, yvec;
	
	abstract public void draw(Graphics raster);	// show calls this every frame
	
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
