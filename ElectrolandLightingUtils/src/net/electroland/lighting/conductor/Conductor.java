package net.electroland.lighting.conductor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.Vector;

import net.electroland.input.InputDeviceEvent;
import net.electroland.input.InputDeviceListener;
import net.electroland.input.devices.HaleUDPInputDevice;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.lighting.tools.SimpleVLM;
import net.electroland.util.OptionException;
import net.electroland.util.Util;

import org.apache.log4j.Logger;

abstract public class Conductor implements InputDeviceListener {

	public final static String ANIMATION_PROPS = "animation.properties";
	public final static String LIGHT_PROPS = "lights.properties";	
	
	private static Logger logger = Logger.getLogger(Conductor.class);
	private Vector<Behavior> behaviors = new Vector<Behavior>();
	private AnimationManager am;
	private DetectorManager dm;
	private HaleUDPInputDevice hs;

	public static Properties getProperties(String resourcename) throws FileNotFoundException, IOException
	{
		Properties props = new Properties();
		InputStream is = new Util().getClass().getClassLoader().getResourceAsStream(resourcename);
		if (is != null)
		{
			props.load(is);
		}else{
			System.out.println("Conductor: failed to find properties file: " + resourcename);
		}
		return props;
	}

	public static URL locateResource(String resourcename) throws FileNotFoundException, IOException
	{
		Enumeration<URL> e = new Util().getClass().getClassLoader().getResources(resourcename);
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
	 * Start animation manager AND lighting manager.
	 * @param propsName
	 */
	final public void initAnimation()
	{
		try {
			dm = new DetectorManager(getProperties(LIGHT_PROPS));
			am = new AnimationManager(dm.getFps(), getProperties(ANIMATION_PROPS));
		} catch (UnknownHostException e) {
			logger.error(e);
		} catch (OptionException e) {
			logger.error(e);
		} catch (FileNotFoundException e) {
			logger.error(e);
		} catch (IOException e) {
			logger.error(e);
		}
	}

	final public void initHaleUDPInputDeviceListener(int port, int bufferLength)
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
		Collections.sort((List)behaviors, new BehaviorComparator());
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

		Iterator<Behavior> i = behaviors.listIterator();
		while (i.hasNext()){			
			Behavior b = i.next();
			b.inputReceived(e);
		}
	}
	
	final public AnimationManager getAnimationManager()
	{
		return am;
	}
	
	final public DetectorManager getDetectorManager()
	{
		return dm;
	}
}