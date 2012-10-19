package net.electroland.gotham.processing.assets;

import net.electroland.gotham.processing.GothamPApplet;
import org.apache.log4j.Logger;
import java.awt.Dimension;
import processing.core.PApplet;

public abstract class Stripe {
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	PApplet p;
	Dimension d;

	public static float scalerAmt;
	public static boolean randomSpeeds = false;
	public static float rScaler;
	public static boolean grow;
	public static float satMin;
	public static float satMax;
	public static float widthMax;

	public float xpos;
	public float targetxpos;
	public int stripeColor;
	public float hue;
	public float originalhue;
	public float saturation = 1;
	public float targetsaturation = 100;
	public float brightness = 100;
	public float w; // with of a stripe
	public float targetw;
	public float widthScaler = 1; //used for the width affecter
	public float wtarget = 1;
	public float targethue;
	public float xoff;
	public float dist;
	public boolean stillEasing, pStillEasing;
	public boolean isNew;
	//used to determine if we need to follow a diff stripe since direction changed.
	public boolean boundaryStripe = false;
	public boolean leftover = false;

	public MoveBehavior movement;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		w = 10;
		stripeColor = ColorPalette.getRandomColor();
	}

	public abstract void setWidth(Stripe s);
	public abstract void setWidth(float num);
	public abstract boolean justFinishedEasing();
	public abstract float getLeftSide();
	public abstract float getRightSide();
	public abstract boolean isOffScreen();
	//public abstract void checkHover(Point2D.Float loc, boolean standing);
	public abstract void checkPinning(PersonMouseSimulator pm);
	public abstract void forcePosition(float p);
	public abstract void update();
	public abstract void display();
	public abstract void display(float n);
	public abstract float getLocation();
	public abstract MoveBehavior getBehavior();
	public abstract void performSaturationShift(float zoneCoord, String value);
	public abstract void performHueShift(float zoneCoord, String value);
	
	public void setBehavior(MoveBehavior mb) {
		movement = mb;
	}
	public boolean isGrowing(){
		if(wtarget > 1.0)
			return true;
		else return false;
	}
	public boolean containsLocation(float test){
		if(test > (xpos-w) && test < xpos+w)
			return true;
		else return false;
	}
	
	public long getSpawnRate(){
		return (long)(((100)/movement.getDist()) * (movement.getTimeAcross()*1000));
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
	public static void setSatMin(float n) {
		satMin = n;
	}
	public static void setSatMax(float n) {
		satMax = n;
	}
	public static void setWidthMax(float n) {
		widthMax = n;
	}
	
}
