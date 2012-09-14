package net.electroland.gotham.processing.assets;

import processing.core.PApplet;
import processing.core.PGraphics;
import de.looksgood.ani.Ani;

public class Stripe {
	public float xpos;
	
	private float h;
	Ani vel;
	private float target;
	private float rate;
	PApplet p;
	PGraphics off;
	public static int scaler = 5;
	public static float w = 50 / scaler;
float pxp;

	public Stripe(PApplet p, PGraphics pg) {
		this.p = p;
		off = pg;
		xpos = -w;
		target = pg.width + w;
		h = p.random(360);
		// rate = random(3, 5); //amt of seconds it would ordinarily take to go
		// full L to R
		rate = 15.0f;
		vel = new Ani(this, rate, "xpos", target, Ani.LINEAR, "onEnd:kill");
	}

	public void run() {
		off.beginDraw();

		// if(mouseDist() < w*5.5){
		float s = PApplet.map(mouseDist(), 300, 0, 90, 0);
		s = PApplet.constrain(s, 0, 90);
		off.fill(p.color(h, s, 90));
		// }
		// else off.fill(color(h, 90,90));

		// off.fill(c);
		off.rect(xpos, off.height / 2, (w * 2) + 2, off.height);
		off.endDraw();
	}

	public boolean okgo() {
		if (xpos >= w && pxp < w) {
			return true;
		} else {
			pxp = xpos;
			return false;
		}
	}

	public boolean kill() {
		if (xpos >= target) {
			return true;
		} else
			return false;
	}

	public float mouseDist() {
		return PApplet.dist(p.mouseX, 0, xpos * scaler, 0);
	}
}

