package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import java.awt.geom.Point2D;

public class StripeFlexRight extends Stripe {

	private boolean pstanding = true;

	public StripeFlexRight(PApplet p, Dimension d) {
		super(p, d);
		movement = new MoveRight(p, d);
	}

	@Override
	public void update() {
		movement.move();
		xpos = movement.getPosition();
	}

	@Override
	public void display() {
		float h = p.hue(stripeColor);
		float v = p.brightness(stripeColor);
		p.fill(h, saturation, v);
		// p.stroke(p.color(0,100,100));
		p.rect(xpos, -25, w+20, p.height + 50);
	}

	@Override
	public void setWidth(Stripe inFront) {
		if (!(movement instanceof Pin))
			w = inFront.xpos - this.xpos;
		else {
			if (grow)
				w = inFront.xpos - this.xpos;
		}
	}

//	@Override
//	public void checkHover(Point2D.Float loc, boolean standing) {
		//
		// if (loc.getX() > xpos && loc.getX() < (xpos + w)) {
		// if (standing && !pstanding)
		// setBehavior(new Pin(p, d, xpos));
		// if (!standing && pstanding) {
		// // setBehavior(new Move(p, d, xpos, target));
		// if (w >= 300)
		// setBehavior(new SpringLeft(p, d, xpos, movement.getTarget()));
		// //something like that
		// else
		// setBehavior(new MoveLeft(p, d));
		// }
		// }
		//
		// pstanding = standing;
//	}

	private Point2D.Float loc = new Point2D.Float(0,0);
	private Point2D.Float ploc = new Point2D.Float(0,0);

	@Override
	public void checkPinning(PersonMouseSimulator pm) {
		loc = pm.getLocation();

		if (loc.getY() >= d.height - 50
				&& ploc.getY() < d.height - 50
				&& movement instanceof MoveRight) {
			if (pm.getZone() >= xpos && pm.getZone() <= (xpos + w))
				setBehavior(new Pin(p, d, xpos));
		}
		if (movement instanceof Pin) {
			if ((loc.getY() < d.height - 50 && ploc.getY() >= d.height - 50)
					|| pm.zoneChanged()) {

				if (Math.abs(w) >= 300) {
					if (Stripe.scalerAmt < 0)
						setBehavior(new SpringLeft(p, d, xpos));
					else
						setBehavior(new SpringRight(p, d, xpos));
				} else {
					if (Stripe.scalerAmt < 0)
						setBehavior(new MoveLeft(p, d, xpos));
					else
						setBehavior(new MoveRight(p, d, xpos));
				}
			}
		}
		ploc.setLocation(loc);
	}
	

	@Override
	public float getLocation() {
		return xpos;
	}

	@Override
	public boolean isOffScreen() {
		return (Math.abs(xpos - movement.getTarget()) < 10 || Math.abs(this.w) < 2);
	}

	@Override
	public void forcePosition(float tx) {
		// setBehavior(new MoveLeft(this.p, this.d, tx, target));
		movement.setPosition(tx);
		// movement = new MoveRight(this.p, this.d, tx);
	}

	@Override
	public MoveBehavior getBehavior() {
		return movement;
	}

}
