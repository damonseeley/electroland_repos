package net.electroland.memphis.core;

import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.events.HaleUDPInputDeviceEvent;
import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;

public class BridgeState extends Behavior {

	private long duration;
	private Bay[] bays;
	
	public BridgeState(long duration, int totalBays){
		bays = new Bay[totalBays];
		for (int i = 0; i < bays.length; i++){
			bays[i] = new Bay("bay " + i);
		}
		this.duration = duration;
	}	

	public void inputReceived(InputDeviceEvent e) {

		byte[] data = ((HaleUDPInputDeviceEvent)e).getData();

		for (int i = 0; i < bays.length; i++)
		{
			if (data[i] == (byte)253){
				bays[i].hit();
			}
		}
	}

	public int getSize(){
		return bays.length;
	}

	public long getTimeSinceLast(int bay)
	{
		return bays[bay].getTimeSinceLast();
	}

	public int getHitCount(int bay)
	{
		return bays[bay].getHitCount();
	}
	
	
	protected class Bay{

		protected String id;
		private ConcurrentLinkedQueue<Long> q = new ConcurrentLinkedQueue<Long>();
		private long last;

		protected Bay(String id)
		{
			this.id = id;
		}
		
		protected void hit()
		{
			synchronized(q){
				this.last = System.currentTimeMillis();
				q.add(last);
			}
		}
		
		protected long getTimeSinceLast()
		{
			if (last == 0)
				return -1;
			else
				return System.currentTimeMillis() - last;
		}
		
		protected int getHitCount()
		{
			synchronized (q){
				while (true){
					Long h = q.peek();
					if (h != null && h < System.currentTimeMillis() - duration)
					{
						q.remove(h);
					}else{
						break;
					}
				}
				return q.size();
			}
		}
	}


	public void completed(Animation a) {
		// meh.
	}
}