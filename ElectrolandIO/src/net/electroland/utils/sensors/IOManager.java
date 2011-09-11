package net.electroland.utils.sensors;

import java.net.InetAddress;

public class IOManager implements Runnable {

	private Thread thread;
	
	public void init()
	{
		init("sensor.properties");
	}
	
	public void init(String propsFileName)
	{
		
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		
	}

	public void start()
	{
		
	}

	public void stop()
	{
		
	}

	public IOState getState(String id)
	{
		return null;
	}

	public IOState[] getStates(String tag)
	{
		return null;
	}

	public IOState[] getStatesForDevice(String deviceName)
	{
		return null;
	}
	
	public IOState[] getStatesForIP(InetAddress ip)
	{
		return null;
	}
}