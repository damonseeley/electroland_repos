package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.EastBlurTest;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;

public class Stripe {

	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private ElectrolandProperties props = GothamConductor.props;

	private EasingFunction ef;
	public float begin;
	public static float target;
	private float prevMillis;
	private float percentComplete;
	private float baseTime;
	public float timeAcross; // Number of seconds needed to traverse the sync
								// area, based off defaultTime

	PApplet p;
	private float xpos;
	//private float h; // the hue of this Stripe
	int stripeColor;
	private float w; // with of a stripe
	private float hw; // half width
	public float dist;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		ef = new Linear();
		w = 30 + (float) Math.random() * 120;
		hw = w * 0.5f;
		begin = -100-hw; // start offscreen
		target = d.width + 200; // end offscreen
		//h = p.random(360);
		stripeColor = EastBlurTest.stripeColors[(int)(Math.random()*EastBlurTest.stripeColors.length)];
		prevMillis = p.millis(); // Start our timer
		dist = target - begin;
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = EastBlurTest.randomSpeeds ? baseTime / 2f
				+ (float) Math.random() * 60 : baseTime * (dist / d.width);
	}

	// Overloaded Constructor. Only used to fill the screen with Stripe on
	// startup.
	public Stripe(PApplet p, Dimension d, int spacer) {
		this(p, d);
		w = d.width / 6;
		hw = w * 0.5f;
		begin = (-100-hw) + spacer * w;
		dist = target - begin;
		baseTime = props.getOptionalInt("wall", "East", "baseTime");
		timeAcross = EastBlurTest.randomSpeeds ? baseTime / 2f
				+ (float) Math.random() * 60 : baseTime * (dist / d.width);
	}

	public void run() {
		float inc = ((p.millis() - prevMillis) / (timeAcross*1000)) * EastBlurTest.scalerAmt;
		percentComplete += inc;
		prevMillis = p.millis();

		xpos = ef.valueAt(percentComplete, begin, target);
		//p.fill(p.color(h, 90, 90));
		p.fill(stripeColor);
		p.rect(xpos, p.height / 2, w*1.3f, p.height + 50); //w and h are exagerated for the benefit of blurring.
	}

	public boolean isOffScreen() {
		return xpos >= target;
	}

	// Only used if we want different sized Stripes to file in perfectly one
	// after the other.
	public float getSpawnRate() {
		// (this stripe's width : total distance to travel) * time to get to the
		// other side
		return ((w / dist) * (timeAcross*1000));
	}

}
