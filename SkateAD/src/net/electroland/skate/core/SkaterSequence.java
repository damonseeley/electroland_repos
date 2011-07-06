package net.electroland.skate.core;

import java.util.List;
import java.util.Vector;

import org.apache.log4j.Logger;

public class SkaterSequence {

	static Logger logger = Logger.getLogger(SkaterSequence.class);
	
	private Vector<SkaterSequenceStep> steps = new Vector<SkaterSequenceStep>();
	private int currLoop = 0, defaultLoops;
	private SkaterSequence nextShow;
	private String name;
	private long startTime;

	public int getDefaultLoops() {
		return defaultLoops;
	}
	
	public double getElapsedInSeconds() {
		return (System.currentTimeMillis() - startTime)/1000.0;
	}

	public void setDefaultLoops(int defaultLoops) {
		this.defaultLoops = defaultLoops;
	}

	public SkaterSequence getNextShow() {
		return nextShow;
	}

	public void setNextShow(SkaterSequence nextShow) {
		this.nextShow = nextShow;
	}

	public Vector<SkaterSequenceStep> getSteps() {
		return steps;
	}

	public String getName() {
		return name;
	}

	public SkaterSequence(String name)
	{
		this.name = name;
	}

	public void startSequence()
	{
		startTime = System.currentTimeMillis();
		logger.info("starting sequence " + this.name + " loop #" + (currLoop + 1));					
	}

	/** yuck.  
	 * 
	 * @param time
	 * @return EITHER a list of startable Skaters OR nothing OR the next show.
	 */
	public List<SkaterSequenceStep> getStartable(long time)
	{
		Vector<SkaterSequenceStep> startable = new Vector<SkaterSequenceStep>();
		long elapsedTime = time - startTime;

		for (SkaterSequenceStep s : steps){
			if (!s.started && elapsedTime > s.delay){
				s.started = true;
				startable.add(s);
			}
		}
		
		return startable;
	}

	
	public SkaterSequence getCurrentSequence()
	{
		boolean everythingStarted = true;
		for (SkaterSequenceStep s : steps)
		{
			everythingStarted = everythingStarted && s.started;
		}
		
		if (everythingStarted){
			logger.info("finished sequence " + this.name + " loop #" + (currLoop + 1));
			resetSteps();
			currLoop++;
			if (currLoop == defaultLoops){
				currLoop = 0; // reset for future runs.
				resetSteps(); // reset steps
				if (nextShow != null){
					nextShow.startSequence(); // initialize the next run
				}
				return nextShow;
			}else{
				// reset for next loop.
				resetSteps(); // reset steps
				startSequence(); // initialize the next run
				return this;
			}
		}else{
			return this;
		}
	}
	
	private void resetSteps(){
		for (SkaterSequenceStep s : steps){
			s.started = false;
		}
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer("SkaterSequence[");
		sb.append("name=").append(name).append(',');
		sb.append("loops=").append(defaultLoops).append(',');
		sb.append("nextShow=");

		if (nextShow != null)
			sb.append(nextShow.name);

		sb.append(',');
		sb.append("steps=").append(steps).append(']');
		return sb.toString();
	}
}