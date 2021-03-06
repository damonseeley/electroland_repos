package net.electroland.gotham.processing.assets;

import java.awt.Dimension;

import net.electroland.gotham.processing.GothamPApplet;

import org.apache.log4j.Logger;

import processing.core.PApplet;

public class StripeSimple {

	static Logger logger = Logger.getLogger(GothamPApplet.class);

	public static float scalerAmt;
	public static boolean randomSpeeds;
	public static float rScaler;
	public static float spawnScaler;
	
	public float begin;
	public float target;
	public float timeAcross; // Number of seconds needed to traverse the sync
								// area, based off defaultTime

	PApplet p;
	private float xpos;
	int stripeColor;
	private float w; // with of a stripe
	public float dist;
	public static boolean changing = false;

	public StripeSimple(PApplet p, Dimension d, int x) {
		this.p = p;
		xpos = x;
		w = (30 + (float) Math.random()*120);

		stripeColor = ColorPalette.getRandomColor();
	}


	public void run() {

		p.rectMode(PApplet.CENTER);
		p.fill(stripeColor);
		p.rect(xpos, p.height / 2, w, p.height + 50);

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
