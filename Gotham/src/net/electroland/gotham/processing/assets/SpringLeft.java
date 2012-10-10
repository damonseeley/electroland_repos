package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.easing.QuadraticOut;
//import net.electroland.ea.easing.DelayedJump;

public class SpringLeft extends MoveLeft {

	public SpringLeft(PApplet p, Dimension d, float begin) {
		super(p, d, begin);
		ef = new QuadraticOut();
		//ef = new DelayedJump();
	}
}
