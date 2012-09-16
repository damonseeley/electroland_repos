package net.electroland.gotham.processing;

import java.awt.Dimension;

import org.apache.log4j.Logger;

import controlP5.Bang;
import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;

public class StripeTestPApplet extends GothamPApplet {

	private static final long serialVersionUID = 449793686955037868L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;

	public int numStripes = 1;
	public float inc;
	public int[] hues = new int[20];
	private final int offset = 20; // A little bleed to account for the black
									// stripes between faded stripes.

	// Select Horizontal or Vertical stripes
	private boolean horizontal = true;
	private float blurAmt = 1;

	private ControlP5 control;
	private ControlWindow window;
	private Controller<Toggle> tg;
	private Controller<Knob> blurKnob;
	private Controller<Knob> stripeKnob;
	private Controller<Bang> colorButton;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		control = new ControlP5(this);

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 200, 200)
				.hideCoordinates().setBackground(color(90));
		// Init num stripes knob
		stripeKnob = control.addKnob("numStripes").setRange(1, 20).setMin(1)
				.setMax(20).setValue(3).setPosition(20, 100).setRadius(30)
				.setNumberOfTickMarks(20).setTickMarkLength(1)
				.snapToTickMarks(true).setColorForeground(color(255))
				.setColorBackground(color(0, 160, 100))
				.setColorActive(color(255, 255, 0))
				.setDragDirection(Knob.HORIZONTAL);

		// Init toggle switch
		tg = control.addToggle("horizontal").setPosition(10, 40)
				.setSize(50, 20).setValue(true).setMode(ControlP5.SWITCH);
		colorButton = control.addBang("setNewColors").setPosition(100, 40)
				.setSize(20, 20).setTriggerEvent(Bang.RELEASE);
		// Init blur knob
		blurKnob = control.addKnob("blurAmt").setRange(1, 100).setValue(10)
				.setPosition(100, 100).setRadius(30)
				.setColorForeground(color(255))
				.setColorBackground(color(0, 160, 100))
				.setColorActive(color(255, 255, 0))
				.setDragDirection(Knob.HORIZONTAL);
		// Set controls to window object
		((Toggle) tg).moveTo(window);
		((Knob) blurKnob).moveTo(window);
		((Knob) stripeKnob).moveTo(window);	//Apparently these are depricated but they work.
		((Bang) colorButton).moveTo(window); //This is the latest release of Cp5. Can't find any other method.

		// Store random hues... Not efficient but fine for now.
		colorMode(HSB, 360, 100, 100);
		for (int i = 0; i < hues.length; i++)
			hues[i] = color((float) Math.random() * 360, 90, 90);
	}

	@Override
	public void drawELUContent() {
		background(0);
		inc = horizontal ? syncArea.width / numStripes : syncArea.height
				/ numStripes;

		for (int i = 0; i < numStripes; i++) {
			fill(hues[i]);
			if (horizontal)
				rect(i * inc - offset, 0, inc + offset * 2, syncArea.height*2);
			else
				rect(0, i * inc - offset, syncArea.width, inc + offset * 2);
		}

		// int blurAmt = useMouse ? floor(map(mouseX, 0, width, 0, BLUR_MAX))
		// : BLUR_AMT;

		loadPixels();
		fastBlur(pixels, (int) blurAmt);
		updatePixels();
	}

	public void setNewColors() {
		for (int i = 0; i < hues.length; i++)
			hues[i] = color((float) Math.random() * 360, 90, 90);
	}

	// ==================================================
	// Super Fast Blur v1.1, reworked by Michael Kontopoulos
	// originally by Mario Klingemann
	// http://incubator.quasimondo.com/processing/superfast_blur.php
	// ==================================================
	private void fastBlur(int[] img, int radius) {

		if (radius < 1) {
			return;
		}
		int w = width;
		int h = height;
		int wm = w - 1;
		int hm = h - 1;
		int wh = w * h;
		int div = radius + radius + 1;
		int r[] = new int[wh];
		int g[] = new int[wh];
		int b[] = new int[wh];
		int rsum, gsum, bsum, x, y, i, p, p1, p2, yp, yi, yw;
		int vmin[] = new int[max(w, h)];
		int vmax[] = new int[max(w, h)];
		int dv[] = new int[256 * div];
		for (i = 0; i < 256 * div; i++) {
			dv[i] = (i / div);
		}

		yw = yi = 0;

		for (y = 0; y < h; y++) {
			rsum = gsum = bsum = 0;
			for (i = -radius; i <= radius; i++) {
				p = img[yi + min(wm, max(i, 0))];
				rsum += (p & 0xff0000) >> 16;
				gsum += (p & 0x00ff00) >> 8;
				bsum += p & 0x0000ff;
			}
			for (x = 0; x < w; x++) {

				r[yi] = dv[rsum];
				g[yi] = dv[gsum];
				b[yi] = dv[bsum];

				if (y == 0) {
					vmin[x] = min(x + radius + 1, wm);
					vmax[x] = max(x - radius, 0);
				}
				p1 = img[yw + vmin[x]];
				p2 = img[yw + vmax[x]];

				rsum += ((p1 & 0xff0000) - (p2 & 0xff0000)) >> 16;
				gsum += ((p1 & 0x00ff00) - (p2 & 0x00ff00)) >> 8;
				bsum += (p1 & 0x0000ff) - (p2 & 0x0000ff);
				yi++;
			}
			yw += w;
		}

		for (x = 0; x < w; x++) {
			rsum = gsum = bsum = 0;
			yp = -radius * w;
			for (i = -radius; i <= radius; i++) {
				yi = max(0, yp) + x;
				rsum += r[yi];
				gsum += g[yi];
				bsum += b[yi];
				yp += w;
			}
			yi = x;
			for (y = 0; y < h; y++) {
				img[yi] = 0xff000000 | (dv[rsum] << 16) | (dv[gsum] << 8)
						| dv[bsum];
				if (x == 0) {
					vmin[y] = min(y + radius + 1, hm) * w;
					vmax[y] = max(y - radius, 0) * w;
				}
				p1 = x + vmin[y];
				p2 = x + vmax[y];

				rsum += r[p1] - r[p2];
				gsum += g[p1] - g[p2];
				bsum += b[p1] - b[p2];

				yi += w;
			}
		}
	}

}
