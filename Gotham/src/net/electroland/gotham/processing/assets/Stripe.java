package net.electroland.gotham.processing.assets;
import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import net.electroland.gotham.processing.East_BlurTest;

public class Stripe {

	private EasingFunction ef;
	public float begin;
	public static float target;
	private float prevMillis;
	private float percentComplete;
	public float timeAcross; // Number of seconds needed to traverse the sync area
	
	PApplet p;
	private float xpos;
	private float h; //the hue of this Stripe
	private float w; // with of a stripe
	private float hw; // half width
	public float dist;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		ef = new Linear();
		w = 30 + (float)Math.random()*120;
		hw = w * 0.5f;
		begin = -hw; //start offscreen
		target = d.width + 150; //end offscreen
		h = p.random(360);
		prevMillis = p.millis(); //Start our timer
		dist = target - begin;
		timeAcross = East_BlurTest.randomSpeeds ? 30.0f + (float)Math.random()*90 : 60.0f * (dist/d.width);
	}
	//Overloaded Constructor. Only used to fill the screen with Stripe on startup.
	public Stripe(PApplet p, Dimension d, int inc){
		this(p, d);
		w = d.width/6;
		hw = w * 0.5f;
		begin = (-hw) + inc * w;
		dist = target - begin;
		timeAcross = East_BlurTest.randomSpeeds ? 30.0f + (float)Math.random()*90 : 60.0f * (dist/d.width);
	}
	
	public void run() {
		float elapsed = p.millis()-prevMillis;
		if(elapsed < timeAcross*1000)
			percentComplete = elapsed / (timeAcross*1000);
		
		xpos = ef.valueAt(percentComplete, begin, target);
		p.fill(p.color(h, 90, 90), 200);
		p.rect(xpos, p.height / 2, w, p.height+50);	
	}
	public boolean isOffScreen() {
		//It doesn't exactly reach the  final target position, dpending on the tween duration vs. framerate.
		return xpos >= target-50; 
	}
	//Only used if we want different sized Stripes to file in perfectly one after the other.
	public float getSpawnRate() {
		// (this stripe's width : total distance to travel) * time to get to the
		// other side
		return (w / dist) * (timeAcross * 1000);
	}
	
}
