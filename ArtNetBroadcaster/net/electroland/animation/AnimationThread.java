package net.electroland.animation;

import java.util.Iterator;
import java.util.Vector;

public class AnimationThread extends Thread {

	private boolean running = true;
	private Animation animation;
	private long delay;
	private Vector<AnimationListener> listeners;

	public AnimationThread(Animation animation, int fps){
		this.animation = animation;
		this.setFPS(fps);
		listeners = new Vector<AnimationListener>();
	}

	final public void stopAndCleanUp(){
		running = false;
	}

	final public void setFPS(int fps){		
		this.delay = (long)(1000.0 / fps);
	}

	final public void addListener(AnimationListener listener){
		listeners.add(listener);
	}
	
	final public void removeListener(AnimationListener listener){
		listeners.remove(listener);
	}
	
	final public void run()
	{
		animation.initialize();
		long deficit = 0;

		while (running && !animation.isDone())
		{
			long startTime = System.currentTimeMillis();
			
			// need to sync here.
			animation.getFrame();
			
			long cycleDuration = System.currentTimeMillis() - startTime;

			// remove whatever time it took to do this work from the allocated
			// delay.
			long adjDelay = delay - cycleDuration;

			// if it took LONGER than the alloocated delay, carry over the
			// deficiit.
			if (adjDelay < 0)
			{
				deficit -= adjDelay;
				adjDelay = 0;
			}else
			{
				// see if we have any deficit to take care of
				if (deficit > 0)
				{
					// subtract deficit from adjDelay.
					adjDelay = adjDelay - deficit;
					
					// if the deficit was greater than what's left of the
					// allocated delay, set the remainer to deficit.
					if (adjDelay < 0){
						deficit = -1* adjDelay;
						adjDelay = 0;
					}
				}
			}
			
			try 
			{
				Thread.sleep(adjDelay);
			} catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}
		
		animation.cleanUp();

		Iterator <AnimationListener>i = listeners.iterator();
		while (i.hasNext()){
			i.next().animationComplete(animation);
		}
	}
}