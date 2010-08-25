package net.electroland.memphis.core;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.events.HaleUDPInputDeviceEvent;
import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;

public class BridgeState extends Behavior {

	private Bay[] bays;
	private int priority;

	/**
	 * 
	 * @param tripThreshold - The maximum time to wait on a sensor before
	 * 							we think the event is too old to consider worth
	 * 							processing.
	 * 
	 * @param processThreshold - The time out required before you should start
	 * 							a new Sprite for a bay sensor.
	 * 
	 * @param totalBays - how many sensors the data packet represents.
	 * 
	 * @param priority - ignore me.
	 */
	public BridgeState(long tripThreshold, long processThreshold, int totalBays, int priority){
		bays = new Bay[totalBays];
		for (int i = 0; i < bays.length; i++){
			bays[i] = new Bay("bay " + i, tripThreshold, processThreshold);
		}
		this.priority = priority;
	}	

	public void inputReceived(InputDeviceEvent e) {

		byte[] data = ((HaleUDPInputDeviceEvent)e).getData();

		for (int i = 0; i < bays.length; i++)
		{
			if (data[i] == (byte)253){
				bays[i].tripped();
			}
		}
	}

	/**
	 * Use this to figure out if you need to start an animation for any
	 * given bay.
	 * 
	 * @param bay
	 * @return
	 */
	public boolean requiresNewSprite(int bay)
	{
		return bays[bay].requiresNewSprint();
	}

	/** 
	 * Call this after you start a sprite on any bay!
	 * @param bay
	 */
	public void spriteStarted(int bay)
	{
		bays[bay].processed();
	}
	
	
	public int getSize(){
		return bays.length;
	}

	/**
	 * only use this for diagnostic output.
	 * @param bay
	 * @return
	 */
	public long getTimeSinceTripped(int bay)
	{
		return bays[bay].getTimeSinceTripped();
	}

	/**
	 * only use this for diagnostic output.
	 * @param bay
	 * @return
	 */
	public long getTimeSinceProcessed(int bay)
	{
		return bays[bay].getTimeSinceProcessed();
	}

	// not used, but required by interface.
	public void completed(Animation a) {
		// meh.
	}

	// not used, but required by interface.
	public int getPriority() {
		return 0; // meh.
	}

	
	protected class Bay{

		protected String id;
		private long lastTripped = -1;
		private long lastProcessed = -1;
		private long tripThreshold = -1;
		private long processThreshold = -1;

		protected Bay(String id, long tripThreshold, long processThreshold)
		{
			this.id = id;
			this.tripThreshold = tripThreshold;
			this.processThreshold = processThreshold;
		}
		
		protected void tripped()
		{
			this.lastTripped = System.currentTimeMillis();
		}

		protected void processed(){
			this.lastProcessed = System.currentTimeMillis();
		}
		
		protected long getTimeSinceTripped()
		{
			if (lastTripped == -1)
				return -1;
			else
				return System.currentTimeMillis() - lastTripped;
		}

		protected long getTimeSinceProcessed()
		{
			if (lastProcessed == -1)
				return -1;
			else
				return System.currentTimeMillis() - lastProcessed;			
		}

		protected boolean requiresNewSprint()
		{
			long proc = getTimeSinceProcessed();
			// it's been at least X milliseconds since we processed this bay's
			// detector (or we never processed it)...
			if (proc == -1 || proc > processThreshold){
				long last = getTimeSinceTripped();	
				if (last > 0 && last < tripThreshold){
					// and the detector was tripped recently.
					return true;
				}
			}
			return false;
		}
	}
}