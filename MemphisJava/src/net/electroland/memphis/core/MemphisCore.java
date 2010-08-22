package net.electroland.memphis.core;

import java.io.FileNotFoundException;
import java.io.IOException;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.TestBehavior;

import org.apache.log4j.Logger;

public class MemphisCore extends Conductor {

	private static Logger logger = Logger.getLogger(MemphisCore.class);

	final static int MAX_PACKET = 2048;
	final static int LISTEN_PORT = 7474;
	
	public MemphisCore()
	{
		// start the animation and detection mangers
		this.initAnimation();

		// start the Hale UDP Device listeners.
		this.initHaleUDPInputDeviceListener(LISTEN_PORT, MAX_PACKET);
		
		// bridge state object (1 minute in ms, # of sensors)
		BridgeState state = new BridgeState(60 * 1000, 27, 0);
 
		// alert the bridge state any time an event occurs.
		this.addBehavior(state);

		// BridgeFrame
		new BridgeFrame(state, 2000);

		// add a behavior to control animation (that has access to what
		// the latest bridge state is
		this.addBehavior(new TestBehavior(state, 1));

		// use the VLM
		this.showSimpleVLM();
		//this.startSystem(); // headless		
	}
	
	public static void main(String args[])
	{
		new MemphisCore();
	}
}