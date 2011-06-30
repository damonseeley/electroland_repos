package net.electroland.skate.core;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class ChannelPool {

	Queue<Integer> availableChannels;

	public ChannelPool(int channels)
	{
		availableChannels = new ConcurrentLinkedQueue<Integer>();
		for (int i = 0; i < channels; i++){
			availableChannels.add(new Integer(i));
		}
	}
	
	public int getFirstAvailable()
	{
		return availableChannels.size() == 0 ? -1 : availableChannels.remove();
	}

	public void releaseChannel(int channel)
	{
		availableChannels.offer(channel);
	}
}