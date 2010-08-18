package net.electroland.sensor;

import java.util.Iterator;
import java.util.Vector;

abstract public class Sensor {

	private Vector<SensorListener> listeners = new Vector<SensorListener>();
	
	abstract public void startSensing();
	abstract public void stopSensing();
	
	public void addListener(SensorListener l)
	{
		listeners.add(l);
	}
	
	public void removeListener(SensorListener l)
	{
		listeners.remove(l);
	}
	
	public void emptyListeners()
	{
		listeners.setSize(0);
	}

	public void notifyListeners(SensorEvent e)
	{
		Iterator<SensorListener> i = listeners.iterator();
		while (i.hasNext())
		{
			i.next().eventSensed(e);
		}
	}
}