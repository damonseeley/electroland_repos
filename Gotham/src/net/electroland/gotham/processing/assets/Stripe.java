package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;

public class Stripe {

	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private ElectrolandProperties props = GothamConductor.props;
	private Dimension d;

	public static float scalerAmt;
	public static boolean randomSpeeds;
	public static float rScaler;
	public static float spawnScaler;
	
	private EasingFunction ef;
	public float begin;
	public float target;
	private float prevMillis;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the sync
								// area, based off defaultTime

	PApplet p;
	private float xpos;
	// private float h; // the hue of this Stripe
	int stripeColor;
	private float w; // with of a stripe
	private float hw; // half width
	public float dist;
	private int pDirection, direction;
	public static boolean changing = false;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		ef = new Linear();
		w = (30 + (float) Math.random()*120) * spawnScaler;
		hw = w * 0.5f;
		direction = scalerAmt > 0 ? 1 : -1;
		pDirection = direction;
		if (scalerAmt > 0) {
			begin = -100 - hw; // start offscreen
			target = d.width + 200; // end offscreen
		} else {
			begin = d.width + 200;
			target = -100 - hw;
		}

		stripeColor = ColorPalette.getRandomColor();
		prevMillis = p.millis(); // Start our timer
		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = randomSpeeds ? baseTime
				+ p.random(-rScaler, rScaler)
				: baseTime * (dist / d.width);
	}

	// Overloaded Constructor. Only used to fill the screen with Stripe on
	// startup.
	public Stripe(PApplet p, Dimension d, int spacer) {
		this(p, d);
		w = d.width / 6;
		hw = w * 0.5f;

		if (scalerAmt > 0) {
			begin = (-hw) + spacer * w;
			target = d.width + 200;
		} else {
			begin = (-hw) + spacer * w;
			target = -100 - hw;
		}

		dist = Math.abs(target - begin);
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = randomSpeeds ? baseTime
				+ p.random(-rScaler, rScaler)
				: baseTime * (dist / d.width);

	}

	public void run() {
		direction = scalerAmt > 0 ? 1 : -1;
		changing = false;
		
		float inc = ((p.millis() - prevMillis) / (timeAcross * 1000))
				* Math.abs(scalerAmt);
		percentComplete += inc;
		prevMillis = p.millis();

		if (direction != pDirection) { // We've changed direction
			resetDirection(direction);
			changing = true;
		}
		xpos = ef.valueAt(percentComplete, begin, target);

		p.rectMode(PApplet.CENTER);
		p.fill(stripeColor);
		p.rect(xpos, p.height / 2, w, p.height + 50);

		pDirection = direction;
	}

	public boolean isOffScreen() {
		if (scalerAmt > 0)
			return xpos >= target;
		else
			return xpos <= target;
	}

	// Only used if we want different sized Stripes to file in perfectly one
	// after the other.
	public float getSpawnRate() {
		// (this stripe's width : total distance to travel) * time to get to the
		// other side
		return ((w / dist) * (timeAcross * 1000));
	}
	
	private void resetDirection(int dir) {
		begin = xpos;
		percentComplete = 0;
		if (dir > 0) {
			target = d.width + 200;
		} else {
			target = -100 - hw;
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
	public static void setSpawnScaler(float ss){
		spawnScaler = ss;
	}
	
}
