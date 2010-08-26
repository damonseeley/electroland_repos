package net.electroland.memphis.behavior;

import net.electroland.input.InputDeviceEvent;
import net.electroland.lighting.conductor.Behavior;
import net.electroland.lighting.detector.animation.Animation;
import processing.core.PApplet;

public class MemphisProcessingBehavior extends Behavior {
	
	protected PApplet p5;
	
	public MemphisProcessingBehavior(PApplet p5){
		this.p5 = p5;
	}

	public int getPriority() {
		// TODO Auto-generated method stub
		return 0;
	}

	public void inputReceived(InputDeviceEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void completed(Animation a) {
		// TODO Auto-generated method stub
		
	}

}
