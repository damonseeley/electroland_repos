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

}
