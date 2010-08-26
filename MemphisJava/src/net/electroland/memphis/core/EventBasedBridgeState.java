package net.electroland.memphis.core;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.devices.memphis.HaleUDPInputDeviceEvent;

public class EventBasedBridgeState {

	private Bay[] bays;
	
	public EventBasedBridgeState(int length)
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

	public boolean requiresNewSprite(int bay, long threshold)
	{
		return !bays[bay].isProcessed(threshold);
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
		public boolean isProcessed(long processTimeout){
			return Math.abs(lastOn - lastProcessed) > processTimeout && lastProcessed > lastOn;
		}
	}
}