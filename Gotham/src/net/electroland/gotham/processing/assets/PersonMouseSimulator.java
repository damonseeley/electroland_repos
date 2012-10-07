package net.electroland.gotham.processing.assets;

import java.awt.geom.Point2D;
import processing.core.PApplet;

public class PersonMouseSimulator {
	private PApplet p;
	private Point2D.Float position;
	private long timeNotMoving;
	private float px;
	private long startTime = 0;
	private boolean still = false;
	private boolean pstill = false;

	public PersonMouseSimulator(PApplet p) {
		this.p = p;
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

	// For Testing
	public boolean onScreen() {
		if (p.mouseX > 0 && p.mouseX < 700 && p.mouseY > 0
				&& p.mouseY < 500) {
			return true;
		} else
			return false;
	}
}
