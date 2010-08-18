package net.electroland.lighting.conductor;

import java.io.IOException;
import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.Vector;

import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.VisualLightingManager;
import net.electroland.sensor.SensorEvent;
import net.electroland.sensor.SensorListener;
import net.electroland.sensor.sensors.HaleUDPSensor;
import net.electroland.util.OptionException;

abstract public class Conductor implements SensorListener {

	private Vector<Behavior> behaviors = new Vector<Behavior>();
	private AnimationManager am;
	private DetectorManager dm;
	private HaleUDPSensor hs;
	
	/*	Sample (includes a lot of future enablement.
	public Conductor(){
		// enable HaldUDP
		this.initHaleUDP(8080, 2048);
		this.initAnimationManager("depends\\lights.properties");
		//this.initLighting("depends\\lights.properties"); // lighting with no manager
		//this.initRemoteLogging("127.0.0.1"); // some day
		//this.initSound(...); // etc.

		// Behaviors will have access to whatever the conductor has access
		// to.  e.g., if it has lights, it has access to lights.  if
		// it has sounds, it has access to sound.  A behavior takes sensor
		// input and decides how to affect the "show"
		Behavior southBound = new SouthBoundBehavior();
		this.addBehavior(southBound);
		
		Behavior northBound = new NorthBoundBehavior();
		this.addBehavior(northBound);

		this.addAnimation(new SpiralAnimation());
		this.addAnimation(new CircleAnimation());
		this.addTransition(new FadeTransition());

		// control using the visual lighting manager
		this.showVLM();
		// versus headless
		// this.startSystem();
		// this.stopSystem();
	}
*/

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

	/**
	 * 
	 * @throws MissingResourcesException
	 */
	final public void showVLM() throws MissingResourcesException
	{
		if (am== null || dm == null)
		{
			throw new MissingResourcesException("VLM requires calling initAnimationManager() first.");
		}else
		{
			new VisualLightingManager(am, dm);
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
			b.setAnimationManager(am);
			b.setDetectorManager(dm);
			am.addListener(b);
			System.out.println("added behavior " + b);
		} // else throw exception?
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
		System.out.println(e);
		// go through each behavior and tell them the event occurred
		Iterator<Behavior> i = behaviors.iterator();
		while (i.hasNext()){
			i.next().eventSensed(e);
		}
	}
}