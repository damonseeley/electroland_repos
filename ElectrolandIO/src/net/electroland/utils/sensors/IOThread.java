package net.electroland.utils.sensors;

import java.util.Collection;

import net.electroland.utils.sensors.devices.IONode;

public class IOThread implements Runnable {

	protected int rate = 1;
	private Thread thread = null;
	private Collection<IONode> devices;
	private boolean isRunning = false;
	
	public IOThread(Collection<IONode> devices)
	{
		this.devices = devices;
	}
	
	@Override
	public void run() {

		// connect all
		for (IONode device: devices)
		{
			device.connect();
		}

		while (isRunning){

			long start = System.currentTimeMillis();
			
			// sync states
			for (IONode device: devices)
			{
				device.readInput();
				device.sendOutput();
			}
			
			long duration = System.currentTimeMillis() - start;
			long delay = (long)(1000.0/rate);

			if (duration < delay)
				delay -= duration;
			else
				delay = 0;
			
			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		// close all
		for (IONode device: devices)
		{
			device.close();
		}
		thread = null;
	}
	
	public void start()
	{
		isRunning = true;
		thread = new Thread(this);
		thread.start();
	}

	public void stop()
	{
		isRunning = false;
	}
}
