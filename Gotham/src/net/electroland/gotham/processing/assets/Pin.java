package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;

public class Pin implements MoveBehavior {

	PApplet p;
	public float xpos;

	public float dist;

	public Pin(PApplet p, Dimension d, float xpos) {
		this.xpos = xpos;
		this.p = p;	
	}

	@Override
	public void move() {
		//Do nothing.
	}

	@Override
	public float getPosition() {
		return xpos;//ef.valueAt(percentComplete, begin, target);
	}

	@Override
	public void setPosition(float x) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public float getTarget() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float getBegin() {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public String toString(){
		return "Pin";
	}

}
