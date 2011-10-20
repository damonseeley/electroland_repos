package net.electroland.EIA.core;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.log4j.Logger;

public class ChannelPool {

	// this would be a more useful class if it store Channel objects that
	// had some kind of broadly applicable information.  Otherwise, could
	// just put a Queue in SoundController.  Or List for that matter.
	Queue<Integer> availableChannels;
	
	private static Logger logger = Logger.getLogger(ChannelPool.class);

	public static void main(String args[]){
		
		// unit tests
		ChannelPool p = new ChannelPool(3);
		assert p.getFirstAvailable() != -1; // should succeed
		assert p.getFirstAvailable() != -1; // should succeed
		assert p.getFirstAvailable() != -1; // should succeed
		assert p.getFirstAvailable() == -1; // should fail
		p.releaseChannel(2);
		assert p.getFirstAvailable() != -1; // should succeed
		System.out.println("test succeeded.");
	}
	
	public ChannelPool(int channels)
	{
		availableChannels = new ConcurrentLinkedQueue<Integer>();
		for (int i = 1; i <= channels; i++){
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
		logger.info("POOL: successfully released channel " + channel);
	}
}