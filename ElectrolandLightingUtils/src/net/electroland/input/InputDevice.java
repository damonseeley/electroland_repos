package net.electroland.input;

import java.util.Iterator;
import java.util.Vector;

abstract public class InputDevice {

	private Vector<InputDeviceListener> listeners = new Vector<InputDeviceListener>();
	
	abstract public void startSensing();
	abstract public void stopSensing();
	
	public void addListener(InputDeviceListener l)
	{
		listeners.add(l);
	}
	
	public void removeListener(InputDeviceListener l)
	{
		listeners.remove(l);
	}
	
	public void emptyListeners()
	{
		listeners.setSize(0);
	}

	public void notifyListeners(InputDeviceEvent e)
	{
		Iterator<InputDeviceListener> i = listeners.iterator();
		while (i.hasNext())
		{
			i.next().inputReceived(e);
		}
	}
}