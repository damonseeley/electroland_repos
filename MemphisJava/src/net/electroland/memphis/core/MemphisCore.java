package net.electroland.memphis.core;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.TestBehavior;

public class MemphisCore extends Conductor {

	final int MAX_PACKET = 2048;
	final int LISTEN_PORT = 7474;
	
	public MemphisCore()
	{
		// enable HaldUDP and animation
		// listen port and max packet should come from a props file or args
		this.initAnimation("depends\\lights.properties");
		this.initHaleUDPSensor(LISTEN_PORT, MAX_PACKET);
		
		// add a behavior (for now, just one)
		this.addBehavior(new TestBehavior());

		// use the VLM
		this.showSimpleVLM();
	}
	
	public static void main(String args[])
	{
		new MemphisCore();
	}
}