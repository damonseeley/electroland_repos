package net.electroland.memphis.core;

import java.util.Iterator;
import java.util.Vector;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.devices.memphis.HaleUDPInputDeviceEvent;
import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;

public class BridgeState extends Behavior {

	private Bay[] bays;
	private int priority;
	private long tripThreshold, samplePeriod;
	private double standingThreshold;
	
	
	public BridgeState(long tripThreshold, long samplePeriod, double standingThreshold, int totalBays, int priority){
		this.tripThreshold = tripThreshold;
		this.samplePeriod = samplePeriod;
		this.standingThreshold = standingThreshold;
		bays = new Bay[totalBays];
		for (int i = 0; i < totalBays; i++){
			bays[i] = new Bay();
		}
		this.priority = priority;
	}
	
	public BridgeState(int length)
	{
		bays = new Bay[length];
	}
	
	public void inputReceived(InputDeviceEvent e)
	{
		if (e instanceof HaleUDPInputDeviceEvent)
		{
			byte[] data = ((HaleUDPInputDeviceEvent)e).getData();
	
			for (int i = 0; i < bays.length; i++)
			{
				bays[i].event(data[i]);
			}
		}
	}

	public long getTimeSinceTripped(int bay){
		return bays[bay].lastOn == -1 ? -1 : System.currentTimeMillis() - bays[bay].lastOn;
	}

	public long getTimeSinceTrippedOff(int bay){
		return bays[bay].lastOff == -1 ? -1 : System.currentTimeMillis() - bays[bay].lastOff;
	}

	public long getTimeSinceProcessed(int bay){
		return bays[bay].lastProcessed == -1 ? -1 : System.currentTimeMillis() - bays[bay].lastProcessed;
	}

	public boolean isStanding(int bay){
		return bays[bay].isStanding(samplePeriod, standingThreshold);
	}
	
	/**
	 * @param bay
	 * @param threshold - ignored
	 * @return
	 */
	public boolean requiresNewSprite(int bay)
	{
		return bays[bay].readyToProcess(tripThreshold);
	}


	public void spriteStarted(int bay)
	{
		bays[bay].process();
	}

	public boolean isOccupied(int bay)
	{
		return bays[bay].isOn();
	}
	
	public int getSize(){
		return bays.length;
	}
	
	class Bay{

		long lastOn = - 1;
		long lastOff = - 1;
		long lastProcessed = -1;
		
		CopyOnWriteArrayList<event> history;
		

		public Bay(){
			history = new CopyOnWriteArrayList<event>();
		}

		public void event(byte b){
			if (b == (byte)253){
				lastOn = System.currentTimeMillis();
				history.add(new event(true, lastOn));					
			}else{
				lastOff = System.currentTimeMillis();
				history.add(new event(false, lastOff));
			}
		}

		public void process(){
			lastProcessed = System.currentTimeMillis();
		}
		
		// if there is a double tap:
		// off < on < processed < off < on
		// last processed is < on.
		// easiest fix: insist on a delay before the second on counts.
		public boolean readyToProcess(long tripThreshold){
			return System.currentTimeMillis() - lastProcessed > tripThreshold && isOn();
		}
		
		// simple logic (that will break if we don't get a packet) is
		// that it's on if the lastOn event is later than the lastOff event.
		public boolean isOn(){
			return lastOn > lastOff;
		}

		public boolean isStanding(long samplePeriodMs, double threshold){

			// case 0: no history
			if (history.size() == 0){
				return false;
			}

			// remove any stale events.
			pareHistory(samplePeriodMs);

			// missing: should massage out any adjacent same state 
			// state events. (kill the right most one).
			// e.g., if a packet was missed, you'll have two 'on' or 'off' events in a row.
			// that should be pared here. unlikely corner case.
			
			// case 1: only one val in history, that's the current state
			if (history.size()==1){
				return history.get(0).isOn;
			}
			
			// case 2: multiple vals in history starting with 'on' val
			long current = System.currentTimeMillis();
			double totalOn = 0;

			
			// 	0	1	2	3	4
			//+		-				(first element off)	-3 = -3 (missing 1 'on' for total 1 /4)
			//	+			-		(first element on)	4 - 1 = 3 (correct)
			//		-			+	(first element off) -3 +0 = -3 (missing 1 'on' for total 1/4)
			//   	+	-			(first element on)  3 - 2 = 1
			//      -       +   -   (first element off) -3 + 1 = -2 (missing +1 'on' for total 2/4
			
			Iterator<event> i = history.iterator();
			while (i.hasNext()){
				event e = i.next();
				if (e.isOn)
					totalOn += (double)(current - e.time);
				else
					totalOn -= (double)(current - e.time);
			}
			if (totalOn < 1){ // case 3: history started with 'off' val
				totalOn = samplePeriodMs + totalOn;
				System.out.println("invert: " + totalOn);
			}
			System.out.println("totalOn: " + totalOn + " samplePeriodMS: " + samplePeriodMs);
			return (totalOn / samplePeriodMs) > threshold;
		}

		private void pareHistory(long samplePeriodMs){
			long oldestAllowable = System.currentTimeMillis() - samplePeriodMs;
			Iterator<event> i = history.iterator();
			while (i.hasNext()){
				event e = i.next();
				if (e.time < oldestAllowable && history.size() > 1){
					history.remove(e);
				}
			}				
		}

		// store when the last event occurred, and whether it was an 'on' or 'off'.
		class event{
			boolean isOn;
			long time;
			public event(boolean isOn, long time){
				this.isOn = isOn;
				this.time = time;
			}
		}
	}

	public void completed(Animation a) {}
	public int getPriority() {return priority;}
}