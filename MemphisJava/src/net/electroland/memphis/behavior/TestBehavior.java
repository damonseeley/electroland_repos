package net.electroland.memphis.behavior;

import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.sensor.SensorEvent;

public class TestBehavior extends Behavior {

	@Override
	public void eventSensed(SensorEvent e) {
		// this needs to figure out what the relationship is between a 
		// sensor being tripped, and sending out an animation.
		
		// it has these assets at it's finger tips.
		this.getAnimationManager();
		this.getDetectorManger();
		
		// when an event comes in, it should locate where the event is,
		// then probably correlate it with the layout of the fixtures
		// and start animating.
	}

	@Override
	public void completed(Animation a) {
		// no idea if we'll need this.  just let's us know if an animation
		// stopped.
	}
}