package net.electroland.animation;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.DetectorManager;

public class AnimationManager implements Runnable, AnimationListener {

	private DetectorManager dmr;
	private Thread thread;
	private CopyOnWriteArrayList <Animation>live;
	private boolean isRunning = false;
	
	public AnimationManager(DetectorManager dmgr){
		this.dmr = dmr;
	}

	public void startAnimation(Animation a, List <DMXLightingFixture> fixtures){
		a.initialize();
		live.add(a);
	}

	public void startAnimation(Animation a, Transition t, DMXLightingFixture d){
		a.initialize();		
	}
	
	// start all animation (presuming any Animations are in the set)
	public void goLive(){
		isRunning = true;
		if (thread != null){
			thread = new Thread(this);
			thread.start();
		}
	}

	// stop all animation
	public void pause(){
		isRunning = false;
	}

	public void run() {
		while (isRunning){

			// for each animation, get the raster and list of 
			// fixtures upon which each is to be rendered.
			
			// (e.g., iterate through the animations, 
//			Iterator
			
			
			
			// do sleep and sleep adjustment here.
		}
		thread = null;
	}

	public void animationComplete(Animation a) {
		live.remove(a);
	}
}