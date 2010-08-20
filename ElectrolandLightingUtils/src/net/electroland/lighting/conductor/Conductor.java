package net.electroland.lighting.conductor;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.Vector;

import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.SimpleVLM;
import net.electroland.sensor.SensorEvent;
import net.electroland.sensor.SensorListener;
import net.electroland.sensor.sensors.HaleUDPSensor;
import net.electroland.util.OptionException;

abstract public class Conductor implements SensorListener {

	private Vector<Behavior> behaviors = new Vector<Behavior>();
	private AnimationManager am;
	private DetectorManager dm;
	private HaleUDPSensor hs;
	
	final public void startSystem()
	{
		if (am != null){
			am.goLive();
		}
		if (hs != null)
		{
			hs.startSensing();
		}
	}

	final public void stopSystem()
	{
		if (hs != null)
		{
			hs.stopSensing();
		}
		if (am != null){
			am.stop();
		}
		
	}


	final public void showSimpleVLM()
	{
		if (am== null || dm == null)
		{
			throw new MissingResourcesException("VLM requires calling initAnimationManager() first.");
		}else
		{
			new SimpleVLM(am, dm, this);
		}
		
	}
	
	/**
	 * Start Lights with no animation.
	 * @param propsName
	final public void initLighting(String propsName)
	{
		// TBD
	}
	 */

	/**
	 * Start animation manager AND lighting manager.
	 * @param propsName
	 */
	final public void initAnimation(String propsName)
	{
		try {
			dm = new DetectorManager(propsName);
			am = new AnimationManager(dm.getFps());
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (OptionException e) {
			e.printStackTrace();
		}
	}

	final public void initHaleUDPSensor(int port, int bufferLength)
	{
		hs = new HaleUDPSensor(port, bufferLength);
		hs.addListener(this);
	}
	
	final public void addBehavior(Behavior b)
	{
		if (am != null)
		{
			// make the Behavior aware of the dm and am.
			b.setAnimationManager(am);
			b.setDetectorManager(dm);
			// tell the am to send animation complete messages to the
			// behavior
			am.addListener(b);
		}
		behaviors.add(b);
	}
	final public void removeBehavior(Behavior b)
	{
		if (am != null)
			am.removeListener(b);		
		behaviors.remove(b);
	}
	final public void emptyBehaviors()
	{
		if (am != null){
			Iterator<Behavior>i = behaviors.iterator();
			while (i.hasNext())
			{
				am.removeListener(i.next());				
			}
		}
		behaviors.setSize(0);
	}

	@Override
	final public void eventSensed(SensorEvent e) {
		// go through each behavior and tell them the event occurred
		Iterator<Behavior> i = behaviors.iterator();
		while (i.hasNext()){
			i.next().eventSensed(e);
		}
	}
}