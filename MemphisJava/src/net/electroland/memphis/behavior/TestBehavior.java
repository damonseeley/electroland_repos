package net.electroland.memphis.behavior;

import java.awt.image.BufferedImage;

import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.memphis.animation.Throb;
import net.electroland.sensor.SensorEvent;

public class TestBehavior extends Behavior {

	private Recipient bridge;
	private AnimationManager am;
	
	public void eventSensed(SensorEvent e) {
		
		// get the bridge and the animation manager.
		// unfortunately, the am and dm aren't set until after
		// instantiation of this object. (the conductor adds them after
		// you pass it an instantiated Behavior)
		if (bridge == null)
		{
			am = this.getAnimationManager();

			// get a handle on the bridge
			DetectorManager dm = this.getDetectorManger();
			bridge = dm.getRecipients().iterator().next();			
		}

		//System.out.println(e); // show the packet received.

		// for any event, just start a throb.
		if (am.getCurrentAnimation(bridge) == null){ // alternate.
			int width = bridge.getPreferredDimensions().width;
			int height = bridge.getPreferredDimensions().height;
			am.startAnimation(new Throb(new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)),
								bridge);
		}
	}

	public void completed(Animation a) {
	}
}