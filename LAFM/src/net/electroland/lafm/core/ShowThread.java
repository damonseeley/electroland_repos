package net.electroland.lafm.core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;

import net.electroland.artnet.util.NoDataException;
import net.electroland.artnet.util.RunningAverage;
import net.electroland.detector.DMXLightingFixture;
import processing.core.PGraphics;

public abstract class ShowThread extends Thread {

	static Logger logger = Logger.getLogger(ShowThread.class);
	
	final static int LOW = 0;
	final static int MEDIUM = 2;
	final static int HIGH = 4;
	final static int HIGHEST = 6;
	final static int NO_COMPLETE = 0;
	final static int COMPLETE = 1;

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
	private RunningAverage avgProcessing;
	private ShowThread next;
	private ShowThread top;

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
		this.avgProcessing = new RunningAverage(30);
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
		this.avgProcessing = new RunningAverage(30);
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

	/**
	 * Request that this show be immediately followed by next.  Follow on 
	 * shows are stored as a linked list, so you can call chain sequentially
	 * to link multiple shows.
	 * 
	 * Note: no listeners will be alerted of the completion of this thread, so
	 * long as next is non-null AND isRunning == true.
	 * 
	 * @param next
	 */
	final public void chain(ShowThread next){
		
		next.top = (this.top == null) ? this : this.top;

		ShowThread current = this;
		while (current.next != null 
				&& next != this){ // check for circularities.
			current = current.next;
		}
		current.next = next;
	}

	final public void run(){

		DecimalFormat d = new DecimalFormat("####.##");
		logger.info("\t\t" + this.getID() + " started with a target FPS of " + d.format(1000.0/delay));

		while ((System.currentTimeMillis() - startTime < lifespan) && isRunning){

			// synch the raster with every fixture.
			// this is taking 2-3 millis.

			long start = System.currentTimeMillis();
			
			doWork(raster);				

			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				i.next().sync(raster);
			}

			avg.markFrame(); // for measuring fps
			// to here.

			try {
				long adjDelay = delay - (System.currentTimeMillis() - start);

				avgProcessing.addValue((double)(System.currentTimeMillis() - start));
				
				if (adjDelay < 0){
					logger.warn("warning:\tprocessing is taking longer than available sleep cycles");
					adjDelay = 0;
				}
				Thread.sleep(adjDelay);					
			} catch (InterruptedException e) {
				logger.error(e.getMessage(), e);
			}
			
			
		}		

		avg.markFrame(); // for measuring fps
		
		try {
			logger.info("\t\t" + this.getID() + " ended with and average FPS of " + d.format(avg.getFPS()));
			logger.debug("\t\t" + this.getID() + " ended with and average frame processing time of " + d.format(avgProcessing.getAvg()));
		} catch (NoDataException e) {
			logger.error(e.getMessage(), e);
		}

		// if no show is chained to this one as a follow on or a force stop
		// has been called on this thread, do the cleanup and notify listeners.
		if (!isRunning || next == null){
			// let the subclass do it's last frame.
			complete(raster);

			// synch the raster with every fixture.
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				i.next().sync(raster);
			}
			
			// tell any listeners that we are done.
			Iterator<ShowThreadListener> j = listeners.iterator();
			while (j.hasNext()){
				j.next().notifyComplete(this.top == null ? this : this.top, 
										(Collection<DMXLightingFixture>)flowers);
			}			
		}else{
			// otherwise, start the follow-on show thread.
			logger.info("chain stop:\t" + this);
			logger.info("chain start:\t" + next);
			// kind of a hack.  we're overwriting the list of fixtures
			// that the follow-on show thread previously contained.  that means
			// chained shows must be ass
			next.flowers = this.flowers;
			next.listeners = this.listeners;
			next.resetLifespan();
			next.start();
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