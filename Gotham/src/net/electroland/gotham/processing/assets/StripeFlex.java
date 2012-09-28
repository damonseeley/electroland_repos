package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.EastFlex;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;

public class StripeFlex {

	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private ElectrolandProperties props = GothamConductor.props;
	private Dimension d;

	private final int offset = 600;
	
	public static float scalerAmt;
	public static boolean randomSpeeds;
	public static float rScaler;
	
	private EasingFunction ef;
	public float begin;
	public float target;
	private float prevMillis;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the sync
								// area, based off defaultTime
	private float localNoiseOffset = 0;

	PApplet p;
	private float xpos;
	// private float h; // the hue of this Stripe
	int stripeColor;
	private float w; // with of a stripe
	private float hw; // half width
	public float dist;
	private int pDirection, direction;
	public static boolean changing = false;
	private int noiseSeed;

	public StripeFlex(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		ef = new Linear();
		
		//w = (30 + (float) Math.random()*120) * spawnScaler;
		w = 50; //Width is based on how far away the guy in front of me is.
		hw = w * 0.5f;
		direction = scalerAmt > 0 ? 1 : -1;
		pDirection = direction;
		if (scalerAmt > 0) {
			begin = -offset - hw; 
			target = d.width + 200;
		} else {
			begin = d.width + 200;
			target = -offset - hw;
		}

		stripeColor = ColorPalette.getRandomColor();
		prevMillis = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = randomSpeeds ? baseTime
				+ p.random(-rScaler, rScaler)
				: baseTime * (dist / d.width);
		noiseSeed = p.frameCount;
	}

	public void run() {
		direction = scalerAmt > 0 ? 1 : -1;
		changing = false;
		localNoiseOffset = 2.1f - p.noise(p.millis()/6000.0f + noiseSeed)*2; //=0;
		
		//Scale the inc by the knob input + the noise offset 
		//If you don't like the effect, you can just delecte localNoiseOffset below.
		float inc = ((p.millis() - prevMillis) / (timeAcross * 1000)) * (Math.abs(scalerAmt)+localNoiseOffset);
		//On top of that, give each individual stripe it's own little bit of noise.
		percentComplete += (inc);
		prevMillis = p.millis();

		if (direction != pDirection) { // We've changed direction
			resetDirection(direction);
			changing = true;
		}
		xpos = ef.valueAt(percentComplete, begin, target);
		display();		
		pDirection = direction;
	}
	
	//TODO: Add some kind of sliders to constrain width MIN and MAX
	public void setWidth(StripeFlex inFront){
		w = (inFront.xpos) - this.xpos; //Lets try out inversion of the stripe width.
		//w = PApplet.constrain( ((inFront.xpos) - this.xpos), 1, 300);
	}
	
	private void display(){
		//p.rectMode(PApplet.CENTER);
		p.fill(stripeColor);
		//p.stroke(p.color(0,100,100));
		p.rect(xpos, -25, 10+(w*direction), p.height + 50);
	}

	public boolean isOffScreen() {
		if (scalerAmt > 0){
			return xpos >= target || w < -1; //Also, remove it if it becomes too small		
		}
		else
			return xpos <= target;
	}
	public void forcePosition(float x){
		this.begin = x;
		w = 10;
	}
	
	private void resetDirection(int dir) {
		begin = xpos;
		percentComplete = 0;
		if (dir > 0) {
			target = d.width + offset;
		} else {
			target = -offset - hw;
		}
		dist = Math.abs(target - begin);
		timeAcross = randomSpeeds ? (baseTime + p.random(
				-rScaler, rScaler))
				* (dist / d.width) : baseTime * (dist / d.width);
		
	}
	
	public static void setScalerAmt(float amt){
		scalerAmt = amt;
	}
	public static void setUseRandomSpeeds(float rs){
		randomSpeeds = rs > 0 ? true : false;
	}
	public static void setRandomScaler(float rscaler){
		rScaler = rscaler;
	}
	
}
