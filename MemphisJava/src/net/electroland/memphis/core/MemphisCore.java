package net.electroland.memphis.core;

import java.io.FileNotFoundException;
import java.io.IOException;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.TestBehavior;

import org.apache.log4j.Logger;

public class MemphisCore extends Conductor {

	private static Logger logger = Logger.getLogger(MemphisCore.class);

	final static String ANIMATION_PROPS = "animation.properties";
	final static String LIGHT_PROPS = "lights.properties";
	final static int MAX_PACKET = 2048;
	final static int LISTEN_PORT = 7474;

	private String aFileLoc, dFileLoc;
	
	public MemphisCore()
	{
		// enable HaldUDP and animation
		// listen port and max packet should come from a props file or args
		try {
			aFileLoc = this.locateResource(ANIMATION_PROPS).toString();
			dFileLoc = this.locateResource(LIGHT_PROPS).toString();

			logger.info("Animation Properties: " + aFileLoc);
			logger.info("Lights Properties: " + dFileLoc);
			
			this.initAnimation(getProperties(ANIMATION_PROPS),
								getProperties(LIGHT_PROPS));

		} catch (FileNotFoundException e) {
			logger.error(e);
		} catch (IOException e) {
			logger.error(e);
		}
		this.initHaleUDPSensor(LISTEN_PORT, MAX_PACKET);

		
		// bridge state object (1 minute in ms, # of sensors)
		BridgeState state = new BridgeState(60 * 1000, 27);
 
		// alert the bridge state any time an event occurs.
		this.addBehavior(state);

		// add a behavior to control animation (that has access to what
		// the latest bridge state is
		this.addBehavior(new TestBehavior(state));

		// BridgeFrame
		new BridgeFrame(state, 2000);

		// use the VLM
		this.showSimpleVLM();
		//this.startSystem(); // headless		
	}
	
	public static void main(String args[])
	{
		new MemphisCore();
	}
}