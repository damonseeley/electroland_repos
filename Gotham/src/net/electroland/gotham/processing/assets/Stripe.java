package net.electroland.gotham.processing.assets;
import net.electroland.gotham.processing.East_BlurTest;
import java.awt.Dimension;
import processing.core.PApplet;
import de.looksgood.ani.Ani;

public class Stripe {
	public float xpos;
	private float h; //the hue of this Stripe
	Ani vel;
	private float target;
	PApplet p;
	public static float w = 100; // with of a stripe
	public static float hw = w * 0.5f; // half width
	private static boolean highlight = true;

	public Stripe(PApplet p, Dimension d) {
		this.p = p;
		xpos = -hw; //start offscreen
		target = d.width + hw; //end offscreen
		h = p.random(360);
		vel = new Ani(this, East_BlurTest.rate, "xpos", target, Ani.LINEAR); //the tween
	}

	public static void setMouseHighlight(boolean b){
		highlight = b ? true : false;
	}
	public void run() {
		float s = highlight ? PApplet.constrain(mouseDist(), 0, 90) : 90;
		p.fill(p.color(h, s, 90));
		p.rect(xpos, p.height / 2, w, p.height);	
	}
	public boolean kill() {
		return xpos >= target;
	}
	private float mouseDist() {
		return Math.abs(p.mouseX-xpos);
	}
}
