package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;

public class Pin implements MoveBehavior {
	private ElectrolandProperties props = GothamConductor.props;

	PApplet p;
	private EasingFunction ef;
	public float begin;
	public float target;
	private long startTime;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; 
	public float xpos;

	public float dist;

	public Pin(PApplet p, Dimension d, float xpos) {
		this.begin = 0;
		this.target = 0;
		this.xpos = xpos;
		this.p = p;
		dist = 0;
		baseTime = 0;
		timeAcross = 0;
		
	}

	@Override
	public void move() {
		//Do nothing.
	}

	@Override
	public float getPosition() {
		return xpos;//ef.valueAt(percentComplete, begin, target);
	}

}
