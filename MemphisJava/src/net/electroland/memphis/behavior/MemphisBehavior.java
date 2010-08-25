package net.electroland.memphis.behavior;

import java.awt.image.BufferedImage;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.input.InputDeviceEvent;
import net.electroland.input.events.HaleUDPInputDeviceEvent;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.AnimationManager;
import net.electroland.memphis.animation.Shooters;
import net.electroland.memphis.animation.Throb;
import net.electroland.memphis.animation.Wave;
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
				int width = bridge.getPreferredDimensions().width;
				int height = bridge.getPreferredDimensions().height;
				//PGraphics pg = p5.createGraphics(width, height, PConstants.P3D);
				//am.startAnimation(new Wave(p5, "depends/wave.properties"), bridge);
				am.startAnimation(new Shooters(p5, "depends/shooters.properties"), bridge);
				//am.startAnimation(new Throb(new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)), bridge);
			}
		}
		
	}

	public void completed(Animation a) {
		// TODO Auto-generated method stub
		
	}

}
