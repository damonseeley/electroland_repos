package net.electroland.memphis.behavior;

import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.sensor.SensorEvent;

public class TestBehavior extends Behavior {

	@Override
	public void eventSensed(SensorEvent e) {
		// start animation
	}

	@Override
	public void completed(Animation a) {
	}
}