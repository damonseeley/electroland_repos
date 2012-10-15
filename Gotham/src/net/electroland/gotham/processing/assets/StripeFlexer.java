package net.electroland.gotham.processing.assets;

import java.awt.Color;
import java.awt.Dimension;
import processing.core.PApplet;
import java.awt.geom.Point2D;

import net.electroland.gotham.processing.FlexingStripes;

public class StripeFlexer extends Stripe {

	// private boolean pstanding = true;
	private float h;
	public float r, g, b, oldr, oldg, oldb;
	public float targetr, targetb, targetg;
	public boolean justFinished;

	public StripeFlexer(PApplet p, Dimension d) {
		super(p, d);
		if (scalerAmt >= 0)
			movement = new MoveRight(p, d);
		else
			movement = new MoveLeft(p, d);
		h = p.random(300, 500);
		
		hue = p.hue(stripeColor);
		saturation = p.saturation(stripeColor);
		brightness = p.brightness(stripeColor);
		
		originalhue = hue;
		
		int originalColor = p.color(hue, saturation, brightness);
		oldr = ((originalColor >> 16) & 0xFF);
		oldg = ((originalColor >> 8) & 0xFF); //These are 0-255 vals
		oldb = ((originalColor) & 0xFF);
	}
	public StripeFlexer(PApplet p, Dimension d, float startx) {
		this(p, d);
		if (scalerAmt >= 0)
			movement = new MoveRight(p, d, startx);
		else
			movement = new MoveLeft(p, d, startx);
		xpos = movement.getBegin();
	}

	@Override
	public void update() {
		movement.move();
		targetxpos = movement.getPosition();
		xpos += (targetxpos - xpos) * 0.08;
		//xpos = movement.getPosition();
		
		//Added these so I can pause other stripes while one is being inserted dynamically
		if(Math.abs(targetw-w) > 1)
			stillEasing = true;
		else stillEasing = false;
		
		if(!stillEasing && pStillEasing){
			justFinished = true;
		} else justFinished = false;
		
		pStillEasing = stillEasing;
	}
	
	@Override
	public boolean justFinishedEasing(){
		return justFinished;
	}

	@Override
	public void display() {
		w += (targetw - w) * 0.08;
		
		p.fill(hue, saturation, brightness);
		// p.stroke(p.color(200,100,100));
		// p.line(xpos-xoff, 0, xpos-xoff, p.height);
		// p.stroke(p.color(0,100,100));
		p.rect(xpos, -25, w * (widthScaler)+20, d.height+50);
		p.rect(xpos, -25, -w * (widthScaler)-20, d.height+50);

	}
	
	//This is a overload of display() that is used with the WIDTH affecter
	private float offset;
	@Override
	public void display(float targetoff){
		offset += (targetoff - offset) * 0.08;
		w += (targetw - w) * 0.1;
		
		p.fill(hue, saturation, brightness);
		p.rect(xpos+offset, -25, (w * (widthScaler))+10,   d.height+50);
		p.rect(xpos+offset, -25, (-w * (widthScaler))-10,  d.height+50);
	}
	

	@Override
	public void setWidth(Stripe inFront) {
		if(!FlexingStripes.flexMode)
			targetw = 50; 
		else
			targetw = (inFront.getLeftSide() - (this.xpos));
	}
	
	@Override
	public void setWidth(float n){
		targetw = n;
	}

	public void performWidthShift(PersonMouseSimulator pm, String value) {
		String[] vals = PApplet.splitTokens(value, "$");
		int radius = Integer.parseInt(vals[0]);
		float min = Float.parseFloat(vals[1]);
		float max = Float.parseFloat(vals[2]);
		widthShiftFactor = max;
		dist = Math.abs(pm.getZone() - xpos);

		if (pm.getLocation().getY() >= d.height - 50) {

			if (dist <= radius) {
				wtarget = max;
			} else {
				wtarget = min;
			}
		} else
			wtarget = min;

		widthScaler += (wtarget - widthScaler) * 0.08;
	}

	public void performSaturationShift(PersonMouseSimulator pm, String value) {
		String[] vals = PApplet.splitTokens(value, "$");
		int radius = Integer.parseInt(vals[0]);
		float minsat = Integer.parseInt(vals[1]);
		float maxsat = Integer.parseInt(vals[2]);

		// if we go to the bottom 50px of the screen
		if (pm.getLocation().getY() >= d.height - 50) {
			float d = Math.abs(pm.getZone() - (xpos)); // keep this w/2???
			if (d <= radius) {
				targetsaturation = maxsat;
			} else
				targetsaturation = minsat;
		} else
			targetsaturation = minsat;

		saturation += (targetsaturation - saturation) * 0.08;
	}

	public void performHueShift(PersonMouseSimulator pm, String value) {
		String[] vals = PApplet.splitTokens(value, "$");
		int radius = Integer.parseInt(vals[0]);
	    //int howFar = Integer.parseInt(vals[1]);
	    //int newHue = (int)((originalhue + howFar) % 360);

		if (pm.getLocation().getY() >= d.height - 50) {
			float d = Math.abs(pm.getZone() - (xpos)); // keep this w/2???
			if (d <= radius) {
				targetr = Math.abs(255 - oldr);//((newHue >> 16) & 0xFF);
				targetg = Math.abs(255 - oldg);//((newHue >> 8) & 0xFF);
				targetb = Math.abs(255 - oldb);//((newHue) & 0xFF);
				System.out.println(targetr + ", " + targetg + ", " + targetb);
			} else {
				targetr = oldr;
				targetg = oldg;
				targetb = oldb;
			}
		}
		else{
			targetr = oldr;
			targetg = oldg;
			targetb = oldb;
		}

		r += ((targetr - r) * 0.1);
		g += ((targetg - g) * 0.1);
		b += ((targetb - b) * 0.1);
		
		float[] hsb = Color.RGBtoHSB((int)r,(int)g,(int)b, null);
		hue = hsb[0]*360;
		saturation = hsb[1] * 100;
		brightness = hsb[2] * 100;
	}

	private Point2D.Float loc = new Point2D.Float(0, 0);
	private Point2D.Float ploc = new Point2D.Float(0, 0);

	@Override
	public void checkPinning(PersonMouseSimulator pm) {
		loc = pm.getLocation();

		if (loc.getY() >= d.height - 50 && ploc.getY() < d.height - 50
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
	public float getLeftSide() {
		return ((this.xpos) - (this.w));

	}

	@Override
	public float getLocation() {
		return xpos;
	}

	@Override
	public boolean isOffScreen() {
		return (Math.abs(xpos - movement.getTarget()) < 10 || Math
				.abs(this.w * 2) < 2);
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
