package net.electroland.lighting.conductor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Properties;
import java.util.Vector;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.InputDeviceListener;
import net.electroland.input.devices.HaleUDPInputDevice;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.SimpleVLM;
import net.electroland.util.OptionException;

import org.apache.log4j.Logger;

abstract public class Conductor implements InputDeviceListener {

	private static Logger logger = Logger.getLogger(Conductor.class);
	private Vector<Behavior> behaviors = new Vector<Behavior>();
	private AnimationManager am;
	private DetectorManager dm;
	private HaleUDPInputDevice hs;

	public Properties getProperties(String resourcename) throws FileNotFoundException, IOException
	{
		Properties props = new Properties();
		InputStream is = this.getClass().getClassLoader().getResourceAsStream(resourcename);
		if (is != null)
		{
			props.load(is);
		}else{
			logger.warn("failed to find properties file: " + resourcename);
		}
		return props;
	}

	public URL locateResource(String resourcename) throws FileNotFoundException, IOException
	{
		Enumeration<URL> e = this.getClass().getClassLoader().getResources(resourcename);
		if (e.hasMoreElements())
		{
			return e.nextElement();
		}else
		{
			return null;
		}
	}
	
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
	final public void initAnimation(Properties aprops, Properties dprops)
	{
		try {
			dm = new DetectorManager(dprops);
			am = new AnimationManager(dm.getFps(), aprops);
		} catch (UnknownHostException e) {
			logger.error(e);
		} catch (OptionException e) {
			logger.error(e);
		}
	}

	final public void initHaleUDPSensor(int port, int bufferLength)
	{
		hs = new HaleUDPInputDevice(port, bufferLength);
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

	final public void inputReceived(InputDeviceEvent e) {
		// go through each behavior and tell them the event occurred
		Iterator<Behavior> i = behaviors.iterator();
		while (i.hasNext()){
			Behavior b = i.next();
			b.inputReceived(e);
		}
	}
}