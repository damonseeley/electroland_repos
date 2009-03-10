package net.electroland.animation;

import java.util.Collection;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.artnet.util.DetectorManagerJPanel;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;

public class AnimationManager implements Runnable {

	private DetectorManager dmr;
	private DetectorManagerJPanel dmp;
	private Thread thread;
	private CopyOnWriteArrayList <Animation>live;
	private CopyOnWriteArrayList <AnimationListener>listeners;	
	private boolean isRunning = false;
	private long delay;

	public AnimationManager(DetectorManager dmr, int fps)
	{
		this.dmr = dmr;
		this.delay = (long)(1000 / (double)fps);
	}

	public AnimationManager(DetectorManagerJPanel dmp, int fps)
	{
		this.dmr = dmp.getDetectorManager();
		this.dmp = dmp;
		this.delay = (long)(1000 / (double)fps);
		
	}

	final public void startAnimation(Animation a, Collection <DMXLightingFixture> fixtures){
		a.initialize();
		
		// store all the fixtures related to the Animation.
		
		live.add(a);
	}

	final public void startAnimation(Animation a, Transition t, Collection <DMXLightingFixture> fixtures){


		// how the FUCK is this going to work.


	}


	// MAYBE have some short cut methods here like startAnimation(Animation, DMXLightingFixture);
	// (really just wrappers the generate single element fixtures)


	final public void addListener(AnimationListener listener){
		listeners.add(listener);
	}

	final public void removeListener(AnimationListener listener){
		listeners.remove(listener);
	}

	// start all animation (presuming any Animations are in the set)
	final public void goLive(){
		isRunning = true;
		if (thread != null){
			thread = new Thread(this);
			thread.start();
		}
	}

	// stop all animation
	final public void pause(){
		isRunning = false;
	}

	final public void run() {
		long startTime;
		while (isRunning){
			startTime = System.currentTimeMillis();

			// do work

			// a:  call getFrame() on all animations. Create a List per fixture
			//     and store the result of each getFrame() on the proper lists..

			// deal with Transitions...

			// b: for each fixture, composite the frames, and call sync

			// c. (if dmp != null) call repaint() on the JPanel

			// c: figure out if isDone has been called on any Animation.

			// yes?  remove it and alert listeners.

			try 
			{
				long cycleDuration = System.currentTimeMillis() - startTime;
				Thread.sleep(cycleDuration >= delay ? 0 : delay - cycleDuration);
			} catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}
		thread = null;
	}
}