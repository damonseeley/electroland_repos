package net.electroland.gotham.processing.assets;

import java.awt.Dimension;

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;

public class MoveRight implements MoveBehavior {
	private ElectrolandProperties props = new ElectrolandProperties("Gotham-global.properties");;

	private PApplet p;
	public EasingFunction ef;
	private Dimension d;
	public float begin;
	public float target;
	private long startTime;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the syncarea, based off defaultTime
	private float localNoiseOffset = 0;
	public float dist;
	private boolean pause = false;
	float pausedx;

	public MoveRight(PApplet p, Dimension d) {
		this.d = d;
		this.p = p;
		
		begin = -offset;
		target = d.width + offset;
		ef = new Linear();
		

		startTime = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime"); 
		timeAcross = Stripe.randomSpeeds ? (baseTime + p.random(-Stripe.rScaler, Stripe.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
	}
	public MoveRight(PApplet p, Dimension d, float x){
		this(p,d);
		begin = x;
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = Stripe.randomSpeeds ? (baseTime + p.random(-Stripe.rScaler, Stripe.rScaler))* (dist / d.width)
				: baseTime * (dist / d.width);
		startTime = p.millis();
		percentComplete = 0;
	}

	@Override
	public void move() {
		localNoiseOffset = 0;//2.1f - p.noise(p.millis()/6000.0f + noiseSeed)*2;
		
		//Scale the inc by the knob input + the noise offset 
		//If you don't like the effect, you can just delecte localNoiseOffset below.
		if(!pause){
			float inc = ((p.millis() - startTime) / (timeAcross * 1000)) * (Math.abs(Stripe.scalerAmt)+localNoiseOffset);
			//On top of that, give each individual stripe it's own little bit of noise.
			percentComplete += (inc);
			startTime = p.millis();
		}
	}
	
	@Override
	public void pause() {
		pausedx = ef.valueAt(percentComplete, begin, target);
		pause = true;
	}
	@Override
	public void resume() {		
		setPosition(pausedx);
		pause = false;
	}
	@Override
	public boolean pauseState(){
		return pause;
	}
	
	@Override
	public float getPosition(){
		return ef.valueAt(percentComplete, begin, target);
	}
	
	@Override
	public void setPosition(float x){
		begin = x;
		target = d.width+offset;
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
		return "R";
	}
	
	@Override
	public float getDist(){
		return dist;
	}
	
	@Override
	public float getTimeAcross(){
		return timeAcross;
	}

}
