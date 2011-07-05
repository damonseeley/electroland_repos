package net.electroland.skate.core;

public class SkaterSequenceStep {

	protected Skater skater;
	protected int delay;
	protected boolean started = false;
	
	public SkaterSequenceStep(Skater skater, int delay)
	{
		this.skater = skater;
		this.delay = delay;
	}
	public String toString()
	{
		return "SkaterSequenceStep[" + skater.name + "," + delay + "]";
	}
}