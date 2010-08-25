package net.electroland.memphis.behavior;

import processing.core.PApplet;
import net.electroland.input.InputDeviceEvent;
import net.electroland.input.events.HaleUDPInputDeviceEvent;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.memphis.animation.MemphisAnimation;
import net.electroland.memphis.core.BridgeState;

public class MemphisBehavior extends MemphisProcessingBehavior {

	private Recipient bridge;
	private AnimationManager am;
	private BridgeState bs;
	private int priority;
	
	public MemphisBehavior(PApplet p5, BridgeState bs, int priority){
		super(p5);
		this.bs = bs;
		this.priority = priority;
	}

	public int getPriority() {
		return priority;
	}

	public void inputReceived(InputDeviceEvent e) {
		// must grab AM and DM on first input event after behavior is instantiated
		if(!((HaleUDPInputDeviceEvent)e).isValid()){	// if not valid, must be the first dummy event
			if (bridge == null){
				am = this.getAnimationManager();
				// get a handle on the bridge
				DetectorManager dm = this.getDetectorManger();
				bridge = dm.getRecipients().iterator().next();			
			}
			
			if (am.getCurrentAnimation(bridge) == null){ // alternate.
				//int width = bridge.getPreferredDimensions().width;
				//int height = bridge.getPreferredDimensions().height;
				//PGraphics pg = p5.createGraphics(width, height, PConstants.P3D);
				//am.startAnimation(new Wave(p5, "depends/wave.properties"), bridge);
				// BRADLEY: Modifed to pass the bridge state in to the Animation 8/25
				am.startAnimation(new MemphisAnimation(p5, "depends/memphisanimation.properties", bs), bridge);
				//am.startAnimation(new Throb(new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)), bridge);
			}
		}
		
	}

	public void completed(Animation a) {
		// may never be called if we only use one continuous animation
	}

}
