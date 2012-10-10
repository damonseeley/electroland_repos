package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import java.awt.geom.Point2D;
import processing.core.PApplet;

public class PersonMouseSimulator {
	private PApplet p;
	private Dimension d;
	private Point2D.Float position;
	private long timeNotMoving;
	private float px;
	private long startTime = 0;
	private boolean still = false;
	private boolean pstill = false;

	public PersonMouseSimulator(PApplet p, Dimension d) {
		this.p = p;
		this.d = d;
		position = new Point2D.Float();
		startTime = p.millis();
		px = p.mouseX;
	}

	public void update() {
		if (Math.abs(p.mouseX - px) <= 5) {
			still = true;
		} else
			still = false;

		if (still && !pstill) {
			startTime = p.millis();
		}

		timeNotMoving = p.millis() - startTime;

		position.setLocation(p.mouseX, p.mouseY);
		px = p.mouseX;
		pstill = still;
	}

	public boolean standing() {
		if (timeNotMoving > 3000) {
			// System.out.println("Not Moving for " + timeNotMoving);
			return true;
		} else {
			// System.out.println("Moving Again");
			return false;
		}
	}

	public Point2D.Float getLocation() {
		return position;
	}
	
	public float getZone(){
		if(p.mouseX <= d.width/3){
			return d.width/6;
		}
		else if(p.mouseX > d.width/3 && p.mouseX <= (2*d.width/3)){
			return d.width/2;
		} else return 5*d.width/6;
		
	}
	public boolean zoneChanged(){
//		if(p.pmouseX <= d.width/3 && p.mouseX > d.width/3 //A to B
//				|| p.pmouseX >= d.width/3 && p.mouseX < d.width/3 //B to A
//				|| p.pmouseX <= 2*d.width/3 && p.mouseX > 2*d.width/3 //B to C
//				|| p.pmouseX <= 2*d.width/3 && p.mouseX < 2*d.width/3)//C to B
//			return true;
//		else return false;
		return false;
	}

	// For Testing
	public boolean onScreen() {
		if (p.mouseX > 0 && p.mouseX < 700 && p.mouseY > 0
				&& p.mouseY < 500) {
			return true;
		} else
			return false;
	}
}
