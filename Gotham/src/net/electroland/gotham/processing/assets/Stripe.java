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
	
	public final int offset = 600;
	public float xpos;
	int stripeColor;
	public float w; // with of a stripe
	
	MoveBehavior movement;
	public float begin;
	public float target;
	
	
	public Stripe(PApplet p, Dimension d){
		this.p = p;
		this.d = d;
		w = 50;
		//movement = new Move(p, d, begin, target);
		stripeColor = ColorPalette.getRandomColor();
	}
	
	public abstract void setWidth(Stripe s);
	public abstract boolean isOffScreen();
	public abstract void checkHover(Point2D.Float loc, boolean standing);
	public abstract void checkHover(Point2D.Float loc);
	public abstract void forcePosition(float p);
	public abstract void update();
	public abstract void display();
	
}
