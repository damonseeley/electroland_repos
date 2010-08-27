package net.electroland.memphis.core;

import net.electroland.lighting.conductor.Conductor;
import net.electroland.memphis.behavior.MemphisBehavior;

import org.apache.log4j.Logger;

import processing.core.PApplet;

public class MemphisCore extends Conductor {

	private static Logger logger = Logger.getLogger(MemphisCore.class);

	final static int MAX_PACKET = 2048;
	final static int LISTEN_PORT = 1001;
	private PApplet p5;
	
	public MemphisCore()
	{
		// start the animation and detection mangers
		this.initAnimation();

		// start the Hale UDP Device listeners.
		this.initHaleUDPInputDeviceListener(LISTEN_PORT, MAX_PACKET);

		// start weather listener (startupDelay, period)
		this.initWeather(2000, 60000);

		// bridge state object
		//  first argument is the threshold sensor.  SECOND IS NOW THE isStanding THREHOLD

		// NEW:
		// first arg is how long to wait (millis) afte a sensor is tripped before another
		// 'on' should be counted.  e.g., double-tap protection.
		
		// the second value is a sample period (millis) to see if someone is
		// standing there.  it's the period to do a running average for.
		
		// the third is the threshold over which we decide a user is standing there.
		
		// the fourth val is the number of bays.
		
		// the fifth is not used.
		
		BridgeState state = new BridgeState(650, 1500, .9, 27, 0);

		// alert the bridge state any time an event occurs.
		this.addBehavior(state);

		// BridgeFrame (200 is the delay before updating the numbers in the bridge state sidebar)
		new BridgeFrame(state, 200);

		// add a behavior to control animation (that has access to what
		// the latest bridge state is
		//this.addBehavior(new TestBehavior(state, 1));
		p5 = new PApplet();
		p5.init();
		MemphisBehavior mb = new MemphisBehavior(p5, state, 1);
		this.addBehavior(mb);

		// StartupInputDeviceEvent is a fake event for kicking off a non-event
		// based Behavior.
		mb.inputReceived(new StartupInputDeviceEvent());
		
		
		// use the VLM
		this.showSimpleVLM();
		//this.startSystem(); // headless	
	}
	
	public static void main(String args[])
	{
		new MemphisCore();
	}
}