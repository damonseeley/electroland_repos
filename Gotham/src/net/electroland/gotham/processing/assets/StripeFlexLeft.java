package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import java.awt.geom.Point2D;

import net.electroland.gotham.processing.FlexingStripes;

public class StripeFlexLeft extends Stripe {

	private boolean pstanding = true;

	public StripeFlexLeft(PApplet p, Dimension d) {
		super(p, d);
		movement = new MoveLeft(p, d);
	}

	@Override
	public void update() {
		movement.move();
		// TODO fix hardcoding of 1 and 2 below
		float targetoffset = FlexingStripes.widthShift ? PApplet.map(widthScaler, 1, 2, 0, w / 4) : 0;
		xoff += (targetoffset - xoff) * 0.05;

		xpos = movement.getPosition();
	}

	@Override
	public void display() {
		hue = p.hue(stripeColor);
		brightness = p.brightness(stripeColor);
		p.fill(hue, saturation, brightness);
		// p.stroke(p.color(0,100,100));

		p.rect(xpos+xoff, -25, w-20, p.height + 50);
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

	public void performColorShift(PersonMouseSimulator pm, String value){
		String[] vals = PApplet.splitTokens(value, "$");
		int radius = Integer.parseInt(vals[0]);
		float minwidth = Integer.parseInt(vals[1]);
		float maxwidth = Integer.parseInt(vals[2]);
		
		//if we go to the bottom 50px of the screen
		if (pm.getLocation().getY() >= d.height - 50) {
			float d = Math.abs(pm.getZone() - (xpos-w/2)); //keep this w/2???
			if(d <= radius){
				saturation = (int)PApplet.map(d, radius,0, minwidth, maxwidth);
			} else saturation = minwidth;
		}
		else saturation = minwidth;
	}

	private Point2D.Float loc = new Point2D.Float(0, 0);
	private Point2D.Float ploc = new Point2D.Float(0, 0);

	@Override
	public void checkPinning(PersonMouseSimulator pm) {
		loc = pm.getLocation();

		if (loc.getY() >= d.height - 50 && ploc.getY() < d.height - 50
				&& movement instanceof MoveLeft) {
			// getZone returns the centroid of the three "zones" described by
			// Damon
			// We only pin a strip if it's currently on top of the "zone" we're
			// standing in.
			if (pm.getZone() <= xpos && pm.getZone() >= (xpos + w))
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
		// setBehavior(new MoveLeft(this.p, this.d, tx, this.target));
		movement.setPosition(tx);
		// movement = new MoveLeft(this.p, this.d, tx);
	}

	@Override
	public MoveBehavior getBehavior() {
		return movement;
	}

}
