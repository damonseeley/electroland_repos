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
	public static boolean randomSpeeds = false;
	public static float rScaler;
	public static boolean grow;

	public float xpos;
	public int stripeColor;
	public float hue;
	public float saturation = 100;
	public float brightness = 100;
	public float w; // with of a stripe
	public float widthScaler = 1; //used for the width affecter
	public float xoff;

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
	public abstract void performColorShift(PersonMouseSimulator pm, String value);
	public void setBehavior(MoveBehavior mb) {
		movement = mb;
	}
	
	public void performWidthShift(PersonMouseSimulator pm, String value){
		String[] vals = PApplet.splitTokens(value, "$");
		int radius = Integer.parseInt(vals[0]);
		float min = Float.parseFloat(vals[1]);
		float max = Float.parseFloat(vals[2]);
		
		if (pm.getLocation().getY() >= d.height - 50) {
			float d = Math.abs(pm.getZone() - xpos); 
			if(d <= radius){
				widthScaler = PApplet.map(d, radius,0, min, max);
			} else widthScaler = min;
		} else widthScaler = min;
	}

	public static void setWindScaler(float val){
		scalerAmt = PApplet.constrain(val * 3.5f, -3.5f, 3.5f); //If input val is -1 to 1, then this scales it up.
	}


	// Gui Stuff
	public static void setScalerAmtFromKnob(float amt) {
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
