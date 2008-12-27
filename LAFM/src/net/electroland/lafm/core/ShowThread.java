package net.electroland.lafm.core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import net.electroland.artnet.util.NoDataException;
import net.electroland.artnet.util.RunningAverage;
import net.electroland.detector.DMXLightingFixture;
import processing.core.PGraphics;

public abstract class ShowThread extends Thread {

	final static int LOW = 0;
	final static int MEDIUM = 2;
	final static int HIGH = 4;
	final static int HIGHEST = 6;

	private PGraphics raster;
	private long delay;
	private long lifespan;
	private List <DMXLightingFixture> flowers;
	private SoundManager soundManager;
	private long startTime;
	private boolean isRunning = true;
	private List <ShowThreadListener>listeners;
	private String ID; // should rename to avoid confusion with Thread.getId();
	private int showPriority;
	private RunningAverage avg;

	public int getShowPriority() {
		return this.showPriority;
	}

	public ShowThread(DMXLightingFixture flower, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics raster, String ID, int showPriority){ // lifespan is in seconds.
		flowers = Collections.synchronizedList(new ArrayList<DMXLightingFixture>());
		flowers.add(flower);
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		this.delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
		this.raster = raster;
		this.ID = ID;
		this.showPriority = showPriority;
		listeners = Collections.synchronizedList(new ArrayList<ShowThreadListener>());
		this.avg = new RunningAverage(30);
	}

	public ShowThread(List <DMXLightingFixture> flowers, 
					  SoundManager soundManager, 
					  int lifespan, int fps,
					  PGraphics raster, String ID, int showPriority){ // lifespan is in seconds.
		this.flowers = flowers;
		this.soundManager = soundManager;
		this.lifespan = lifespan * 1000;
		this.delay = (long)(1000.0 / fps);
		this.startTime = System.currentTimeMillis();
		this.raster = raster;
		this.ID = ID;
		this.showPriority = showPriority;
		listeners = Collections.synchronizedList(new ArrayList<ShowThreadListener>());
		this.avg = new RunningAverage(30);
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
		this.isRunning = false;
	}
	
	final public String getID(){
		return ID;
	}

	final public void resetLifespan(){
		this.startTime = System.currentTimeMillis();
	}

	final public Collection <DMXLightingFixture> getFlowers(){
		return flowers;
	}

	final public void addListener(ShowThreadListener listener){
		listeners.add(listener);
	}

	final public void removeListener(ShowThreadListener listener){
		listeners.remove(listener);
	}

	final public void run(){

		while ((System.currentTimeMillis() - startTime < lifespan) && isRunning){

			// let the subclass do some work.
			doWork(raster);
			
			// synch the raster with every fixture.
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				i.next().sync(raster);
			}

			avg.markFrame(); // for measuring fps

			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}		
		
		// let the subclass do it's last frame.
		complete(raster);

		// synch the raster with every fixture.
		Iterator <DMXLightingFixture> i = flowers.iterator();
		while (i.hasNext()){
			i.next().sync(raster);
		}

		avg.markFrame(); // for measuring fps
		
		try {
			DecimalFormat d = new DecimalFormat("####.##");
			System.out.println("\t\t" + this.getID() + " ended with and average fps of " + d.format(avg.getFPS()));
		} catch (NoDataException e) {
			e.printStackTrace();
		}
		
		// tell any listeners that we are done.
		Iterator<ShowThreadListener> j = listeners.iterator();
		while (j.hasNext()){
			j.next().notifyComplete(this, (Collection<DMXLightingFixture>)flowers);
		}
	}

	public String toString(){
		StringBuffer sb = new StringBuffer(this.ID);
		sb.append("[");
		Iterator <DMXLightingFixture> i = flowers.iterator();
		while (i.hasNext()){
			sb.append(i.next().getID());
			if (i.hasNext()){
				sb.append(", ");
			}
		}
		sb.append("]");		
		return sb.toString();
	}
}