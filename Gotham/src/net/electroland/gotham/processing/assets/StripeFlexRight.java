package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import java.awt.geom.Point2D;

public class StripeFlexRight extends Stripe {

	private boolean pstanding = true;

	public StripeFlexRight(PApplet p, Dimension d) {
		super(p, d);
		begin = -offset;
		target = d.width + offset;
		movement = new Move(p, d, begin, target);
	}

	@Override
	public void update() {
		movement.move();
		xpos = movement.getPosition();
	}

	@Override
	public void display() {
		// p.rectMode(PApplet.CENTER);
		p.fill(stripeColor);
		// p.stroke(p.color(0,100,100));
		p.rect(xpos, -25, 10 + w, p.height + 50);
	}

	@Override
	public void setWidth(Stripe inFront) {
		w = inFront.xpos - this.xpos;
	}

	@Override
	public void checkHover(Point2D.Float loc, boolean standing) {

		if (loc.getX() > xpos && loc.getX() < (xpos + w)) {
			if (standing && !pstanding)
				setBehavior(new Pin(p, d, xpos));
			if (!standing && pstanding) {
				// setBehavior(new Move(p, d, xpos, target));
				if (w >= 300)
					setBehavior(new Spring(p, d, xpos, target));
				else
					setBehavior(new Move(p, d, xpos, target));
			}
		}

		pstanding = standing;
	}

	private Point2D.Float ploc = new Point2D.Float(0, 0);

	@Override
	public void checkHover(Point2D.Float loc) {

		if (loc.getX() > xpos && loc.getX() < (xpos + w)) {
			if (loc.getY() >= d.height - 50 && ploc.getY() < d.height - 50)
				setBehavior(new Pin(p, d, xpos));
			if ((loc.getY() < d.height - 50 && ploc.getY() >= d.height - 50)
					|| (loc.getX() <= xpos && ploc.getX() > xpos)
					|| (loc.getX() >= xpos + w && ploc.getX() < xpos + w)) {
				if (w >= 300)
					setBehavior(new Spring(p, d, xpos, target));
				else
					setBehavior(new Move(p, d, xpos, target));
			}
		}

		ploc.setLocation(loc);
	}

	@Override
	public boolean isOffScreen() {
		return xpos >= target || w < -1;
	}

	@Override
	public void forcePosition(float tx) {
		setBehavior(new Move(p, d, tx, target));
	}

	public void setBehavior(MoveBehavior mb) {
		movement = mb;
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

}
