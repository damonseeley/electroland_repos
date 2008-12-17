package net.electroland.lafm.core;

import net.electroland.detector.DMXLightingFixture;
import processing.core.PGraphics;

public abstract class ShowThread extends Thread {
	
	private PGraphics raster;
	private long delay;
	private long lifespan;
	private DMXLightingFixture[] flowers;
	private SoundManager soundManager;
	private long startTime;
	private boolean noForce = true;
	
	public ShowThread(DMXLightingFixture flower, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics image){
		flowers = new DMXLightingFixture[1];
		flowers[0] = flower;
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
	}
	
	
	public ShowThread(DMXLightingFixture[] flowers, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics image){ // lifespan is in seconds.
		this.flowers = flowers;
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
	}

	public PGraphics getRaster() {
		return raster;
	}

	public SoundManager getSoundManager() {
		return soundManager;
	}

	abstract public void complete();
	abstract public void doWork();

	final public void forceStop(){
		noForce = false;
	}
	
	final public void resetLifespan(){
		this.startTime = System.currentTimeMillis();		
	}
	
	final public DMXLightingFixture[] getFlowers(){
		return flowers;
	}
	
	final public void addListener(ShowThreadListener listener){
		/**
		 * TODO: add to list of listeners
		 */
	}
	
	final public void removeListener(ShowThreadListener listener){
		/**
		 * TODO: remove from list of listeners
		 */
	}
	
	final public void notifyListeners(){
		/**
		 * TODO: send message to all listeners
		 */
	}
	
	public void run(){
		while ((System.currentTimeMillis() - startTime < lifespan) && noForce){
			
			doWork();
			for (int i = 0; i<flowers.length; i++){
				flowers[i].sync(this.getRaster());
			}
			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		complete();
		notifyListeners();
	}
}
