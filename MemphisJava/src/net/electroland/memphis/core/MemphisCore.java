package net.electroland.memphis.core;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.TestBehavior;

public class MemphisCore extends Conductor {

	final int MAX_PACKET = 2048;
	final int LISTEN_PORT = 8080;
	
	public MemphisCore()
	{
		// enable HaldUDP and animation
		// listen port and max packet should probably come in through
		// either lights.properties or the args in main.
		this.initHaleUDPSensor(LISTEN_PORT, MAX_PACKET);
		this.initAnimation("depends\\lights.properties");
		
		// add a behavior (for now, just one)
		this.addBehavior(new TestBehavior());
		
		// use the VLM
		this.showVLM();

		// start everything
		this.startSystem();
	}
	
	public static void main(String args[])
	{
		new MemphisCore();
	}
}