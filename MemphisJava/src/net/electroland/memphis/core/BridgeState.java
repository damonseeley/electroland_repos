package net.electroland.memphis.core;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.devices.memphis.HaleUDPInputDeviceEvent;
import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;

public class BridgeState extends Behavior {

	private Bay[] bays;
	private int priority;
	
	public BridgeState(long tripThreshold, long processThreshold, int totalBays, int priority){
		bays = new Bay[totalBays];
		for (int i = 0; i < totalBays; i++){
			bays[i] = new Bay(processThreshold);
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
	
	/**
	 * @param bay
	 * @param threshold - ignored
	 * @return
	 */
	public boolean requiresNewSprite(int bay)
	{
		return bays[bay].readyToProcess();
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

		long processTimeout;
		
		public Bay(long processTimeout){
			this.processTimeout = processTimeout;
		}

		public void event(byte b){
			if (b == (byte)253){
				lastOn = System.currentTimeMillis();
			}else{
				lastOff = System.currentTimeMillis();
			}
		}

		public void process(){
			lastProcessed = System.currentTimeMillis();
		}
		
		// simple logic (that will break if we don't get a packet) is
		// that it's on if the lastOn event is later than the lastOff event.
		public boolean isOn(){
			return lastOn > lastOff;
		}

		// if there is a double tap:
		// off < on < processed < off < on
		// last processed is < on.
		// easiest fix: insist on a delay before the second on counts.
		public boolean readyToProcess(){
			return System.currentTimeMillis() - lastProcessed > processTimeout && isOn();
		}
	}

	public void completed(Animation a) {}
	public int getPriority() {return priority;}
}