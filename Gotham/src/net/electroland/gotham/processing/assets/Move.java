package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;

public class Move implements MoveBehavior {
	private ElectrolandProperties props = GothamConductor.props;

	PApplet p;
	public EasingFunction ef;
	public float begin;
	public float target;
	private long startTime;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the syncarea, based off defaultTime
	private float localNoiseOffset = 0;
	private int noiseSeed;
	public float dist;

	public Move(PApplet p, Dimension d, float begin, float target) {
		this.begin = begin;
		this.target = target;
		ef = new Linear();
		this.p = p;

		startTime = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime"); //                                        Test this...
		timeAcross = StripeFlexRight.randomSpeeds ? (baseTime + p.random(-StripeFlexRight.rScaler, StripeFlexRight.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
		noiseSeed = p.millis();
	}

	@Override
	public void move() {
		localNoiseOffset = 2.1f - p.noise(p.millis()/6000.0f + noiseSeed)*2;
		
		//Scale the inc by the knob input + the noise offset 
		//If you don't like the effect, you can just delecte localNoiseOffset below.
		float inc = ((p.millis() - startTime) / (timeAcross * 1000)) * (Math.abs(StripeFlexRight.scalerAmt)+localNoiseOffset);
		//On top of that, give each individual stripe it's own little bit of noise.
		percentComplete += (inc);
		startTime = p.millis();

	}
	@Override
	public float getPosition(){
		return ef.valueAt(percentComplete, begin, target);
	}

}
