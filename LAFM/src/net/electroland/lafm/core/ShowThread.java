package net.electroland.lafm.core;

import java.util.ArrayList;
import java.util.Iterator;

import net.electroland.detector.DMXLightingFixture;
import processing.core.PGraphics;

public abstract class ShowThread extends Thread {

	private PGraphics raster;
	private long delay;
	private long lifespan;
	private DMXLightingFixture[] flowers;
	private SoundManager soundManager;
	private long startTime;
	private boolean noStopRequested = true;
	private ArrayList <ShowThreadListener>listeners;

	public ShowThread(DMXLightingFixture flower, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics raster){ // lifespan is in seconds.
		this.flowers = new DMXLightingFixture[1];
		this.flowers[0] = flower;
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		this.delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
		this.raster = raster;
		listeners = new ArrayList<ShowThreadListener>();
	}

	public ShowThread(DMXLightingFixture[] flowers, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics raster){ // lifespan is in seconds.
		this.flowers = flowers;
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		this.delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
		this.raster = raster;
		listeners = new ArrayList<ShowThreadListener>();
	}

	/**
	 * This will be called at the end of the run cycle if an outside caller
	 * calls forceStop() or if the thread has exceeded it's programmed lifespan.
	 * 
	 * Implement any code you want to happen on the final frame here.
	 */
	abstract public void complete(PGraphics raster);

	/**
	 * Call this per frame to render on the raster.
	 */
	abstract public void doWork(PGraphics raster);

	final public PGraphics getRaster() {
		return raster;
	}

	final public SoundManager getSoundManager() {
		return soundManager;
	}

	final public void cleanStop(){
		this.noStopRequested = false;
	}

	final public void resetLifespan(){
		this.startTime = System.currentTimeMillis();
	}

	final public DMXLightingFixture[] getFlowers(){
		return flowers;
	}

	final public void addListener(ShowThreadListener listener){
		listeners.add(listener);
	}

	final public void removeListener(ShowThreadListener listener){
		listeners.remove(listener);
	}

	final public void run(){
		
		while ((System.currentTimeMillis() - startTime < lifespan) && noStopRequested){

			// let the subclass do some work.
			doWork(raster);

			// first frame is always black.  why?
			
			// synch the raster with every fixture.
//			System.out.println("starting a show thread attached to :");
			for (int i = 0; i<flowers.length; i++){
//				System.out.println("\t" + flowers[i].getID());
				flowers[i].sync(raster);
			}

			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}		
		
		// let the subclass do it's last frame.
		complete(raster);

		// synch the raster with every fixture.
		for (int i = 0; i<flowers.length; i++){
			flowers[i].sync(raster);
		}

		// tell any listeners that we are done.
		Iterator<ShowThreadListener> i = listeners.iterator();
		while (i.hasNext()){
			i.next().notifyComplete(this, flowers);
		}
	}
}