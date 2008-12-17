package net.electroland.connection.animation;

import java.util.Vector;
import java.util.Iterator;

/**
 * This is an abstract implementation of a transition.  You instantiate subclasses
 * of it by passing it the start and end animations.
 * 
 * Then, implement your draw method.  draw can grab (and advance) the pixels of each
 * of the start and end animations using getStartAnimationPixels() and 
 * getEndAnimationPixels.  When your transition is complete, call complete().  
 * Any registered TransitionListeners will be informed that you are done, and
 * can behave accordingly.
 * 
 * @author geilfuss
 *
 */

abstract public class Transition implements Animation{

	private Animation startAnimation, endAnimation;
	private Vector <TransitionListener> listeners;

	public Transition(Animation startAnimation, Animation endAnimation){
		this.startAnimation = startAnimation;
		this.endAnimation = endAnimation;
	}

	final public byte[] getStartAnimationPixels() {
		return startAnimation.draw();
	}

	final public byte[] getEndAnimationPixels(){
		return endAnimation.draw();
	}

	final public Animation getStartAnimation(){
		return startAnimation;
	}
	
	final public Animation getEndAnimation(){
		return endAnimation;		
	}
	
	final public void complete(){
		System.out.println("transition complete");
		startAnimation.stop();
		Iterator <TransitionListener> i = listeners.iterator();
		while (i.hasNext()){
			i.next().transitionComplete(this);
		}
	}

	final public void addListener(TransitionListener listener){
		if (listeners == null){
			listeners = new Vector <TransitionListener>();
		}
		listeners.add(listener);
	}

	final public void removeListener(TransitionListener listener){
		if (listeners != null){
			listeners.remove(listener);
		}
	}
}