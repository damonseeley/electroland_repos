package net.electroland.memphis.core;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.TestBehavior;

public class MemphisCore extends Conductor {

	public MemphisCore()
	{
		// enable HaldUDP and animation
		this.initHaleUDPSensor(8080, 2048);
		this.initAnimation("depends\\lights.properties");
		
		// add a behavior
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