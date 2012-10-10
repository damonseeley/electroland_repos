package net.electroland.gotham.processing.assets;

import net.electroland.gotham.processing.GothamPApplet;
import org.apache.log4j.Logger;
import java.awt.Dimension;
import java.awt.geom.Point2D;
import processing.core.PApplet;

public abstract class Stripe {
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	PApplet p;
	Dimension d;

	public static float scalerAmt;
	public static boolean randomSpeeds;
	public static float rScaler;
	public static boolean grow;

	public float xpos;
	public int stripeColor;
	public float saturation;
	public float w; // with of a stripe

	public MoveBehavior movement;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		w = 50;
		stripeColor = ColorPalette.getRandomColor();
	}

	public abstract void setWidth(Stripe s);
	public abstract boolean isOffScreen();
	//public abstract void checkHover(Point2D.Float loc, boolean standing);
	public abstract void checkPinning(PersonMouseSimulator pm);
	public abstract void forcePosition(float p);
	public abstract void update();
	public abstract void display();
	public abstract float getLocation();
	public abstract MoveBehavior getBehavior();

	public void setBehavior(MoveBehavior mb) {
		movement = mb;
	}
	
	public void performColorShift(PersonMouseSimulator pm, int radius){
		//if we go to the bottom 50px of the screen
		if (pm.getLocation().getY() >= d.height - 50) {
			float d = Math.abs(pm.getZone() - (xpos+w/2)); 
			if(d <= radius){
				saturation = (int)PApplet.map(d, radius,0, 40, 100);
			} else saturation = 40;
		}
		else saturation = 40;
	}
	
	//Hook here?
	public static void setScaler(float norm){
		//scalerAmt = PAppet.map(norm, -1,1, X,Y);
	}


	// Gui Stuff
	public static void setScalerAmt(float amt) {
		scalerAmt = amt;
	}

	public static void setUseRandomSpeeds(float rs) {
		randomSpeeds = rs > 0 ? true : false;
	}

	public static void setRandomScaler(float rscaler) {
		rScaler = rscaler;
	}
	public static void setGrow(float g){
		grow = g > 0 ? true : false;
	}

}
