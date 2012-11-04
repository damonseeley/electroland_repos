package net.electroland.gotham.processing.assets;

import java.awt.Dimension;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;

public class MoveLeft implements MoveBehavior {
	private ElectrolandProperties props = GothamConductor.props;

	PApplet p;
 	Dimension d;
	public EasingFunction ef;
	public float begin;
	public float target;
	private long startTime;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the syncarea, based off defaultTime
	private float localNoiseOffset = 0;
	public float dist;
	
	public final int offset = 600;

	public MoveLeft(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		begin = d.width + offset;
		target = -offset;
		//ef = new Linear();
		//ef = new SinusoidalInOut();
		ef = new Linear();
		

		startTime = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime"); 
		timeAcross = Stripe.randomSpeeds ? (baseTime + p.random(-Stripe.rScaler, Stripe.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
		p.noiseSeed(p.millis());
	}
	public MoveLeft(PApplet p, Dimension d, float x){
		this(p,d);
		begin = x;
		dist = Math.abs(target - begin);
		timeAcross = Stripe.randomSpeeds ? (baseTime + p.random(-Stripe.rScaler, Stripe.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
		startTime = p.millis();
		percentComplete = 0;
	}
	
	@Override
	public void pause() {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void resume() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void move() {
		localNoiseOffset = 0;//2.1f - p.noise(p.millis()/6000.0f + noiseSeed)*2;
		
		//Scale the inc by the knob input + the noise offset 
		//If you don't like the effect, you can just delecte localNoiseOffset below.
		float inc = ((p.millis() - startTime) / (timeAcross * 1000)) * (Math.abs(Stripe.scalerAmt)+localNoiseOffset);
		//On top of that, give each individual stripe it's own little bit of noise.
		percentComplete += (inc);
		startTime = p.millis();

	}
	@Override
	public float getPosition(){
		return ef.valueAt(percentComplete, begin, target);
	}
	
	@Override
	public void setPosition(float x){
		begin = x;
		target = -offset;
		startTime = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		timeAcross = Stripe.randomSpeeds ? (baseTime + p.random(-Stripe.rScaler, Stripe.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
		percentComplete = 0;
	}
	@Override
	public float getTarget() {
		return target;
	}

	@Override
	public float getBegin() {
		return begin;
	}
	
	public String toString(){
		return "L";
	}
	
	@Override
	public float getDist(){
		return dist;
	}
	
	@Override
	public float getTimeAcross(){
		return timeAcross;
	}
	@Override
	public boolean pauseState() {
		// TODO Auto-generated method stub
		return false;
	}

}
