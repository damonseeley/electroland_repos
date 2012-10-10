package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.easing.QuadraticOut;
//import net.electroland.ea.easing.DelayedJump;

public class SpringRight extends MoveRight {

	public SpringRight(PApplet p, Dimension d, float begin) {
		super(p, d, begin);
		ef = new QuadraticOut();
		//ef = new DelayedJump();
	}
}
