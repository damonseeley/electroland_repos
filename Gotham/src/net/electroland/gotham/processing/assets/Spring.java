package net.electroland.gotham.processing.assets;

import java.awt.Dimension;
import processing.core.PApplet;
import net.electroland.ea.easing.QuinticOut;
//import net.electroland.ea.easing.DelayedJump;

public class Spring extends Move {

	public Spring(PApplet p, Dimension d, float begin, float target) {
		super(p, d, begin, target);
		ef = new QuinticOut();
		//ef = new DelayedJump();
	}
}
