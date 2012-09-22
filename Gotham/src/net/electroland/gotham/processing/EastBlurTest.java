package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.Stripe;
import net.electroland.utils.ElectrolandProperties;

import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;

import org.apache.log4j.Logger;

public class EastBlurTest extends GothamPApplet {

	public static boolean randomSpeeds;
	public static float scaler; // A scaler value to upsample or downsample the duration of the tween	
	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	private int nStripes; //Num Stripes that begin on screen.

	private ControlP5 control;
	private ControlWindow window;
	private Controller<Toggle> bgColor;
	private Controller<Toggle> rSpeeds;
	private Controller<Knob> blurKnob;
	private Controller<Knob> speedKnob;
	public float scalerAmt;
	public float blurAmt;
	public boolean blackOrWhite;

	ArrayList<Stripe> stripes;
	private float spawnRate;
	private long startTime = 0;
	public float percentComplete;

	private ElectrolandProperties props = GothamConductor.props;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		rectMode(CENTER);

		stripes = new ArrayList<Stripe>();
		// Populate the screen with several existing stripes.
		nStripes = props.getOptionalInt("wall", "East", "initialStripes");
		scaler = (float)props.getOptionalInt("wall", "East", "initialScaler");
		//randomSpeeds = props.getOptionalBoolean("wall", "East", "randomSpeeds");
		for (int i = nStripes; i >= 0; i--)
			stripes.add(new Stripe(this, syncArea, i));
		// How often to generate a new stripe
		spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
		startTime = millis();
		
		initGui();

		logger.info("Initial OnScreen Stripes: " + nStripes);
		logger.info("Initial Speed Scaler: " + scaler);
	}

	@Override
	public void drawELUContent() {
		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));

		scaler = scalerAmt; //Point the class' scaler val to the knob
		
		// Handle Stripes
		for (int i = stripes.size() - 1; i >= 0; i--) {
			Stripe s = stripes.get(i);
			s.run();
			if (s.isOffScreen())
				stripes.remove(i);
		}

		// Timing Control for each new Stripe
		float inc = ((millis() - startTime) / (spawnRate)) * scaler;
		percentComplete += inc;
		startTime = millis();
		if (percentComplete > 0.98) {
			stripes.add(new Stripe(this, syncArea));
			spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
			percentComplete = 0;
		}

		// Blur. Right now, blur is controlled by the vertical mouse component.
		loadPixels();
		fastBlur(pixels, floor(blurAmt));
		updatePixels();
	}

	private void initGui() {
		control = new ControlP5(this);

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 200, 200)
				.hideCoordinates().setBackground(color(90));
		//Speed Scaler Knob
		speedKnob = control.addKnob("scalerAmt").setRange(0.2f, 3.5f).setValue(1)
						.setPosition(10, 100).setRadius(30)
						.setColorForeground(color(255))
						.setColorBackground(color(200, 160, 100))
						.setColorActive(color(255, 60, 60))
						.setDragDirection(Knob.HORIZONTAL);
		// Init blur knob
		blurKnob = control.addKnob("blurAmt").setRange(1, 100).setValue(5)
						.setPosition(100, 100).setRadius(30)
						.setColorForeground(color(255))
						.setColorBackground(color(200, 160, 100))
						.setColorActive(color(255, 60, 60))
						.setDragDirection(Knob.HORIZONTAL);
		// Init toggle switch
		bgColor = control.addToggle("blackOrWhite").setPosition(10, 40)
						.setSize(50, 20).setValue(true).setMode(ControlP5.SWITCH);
		rSpeeds = control.addToggle("randomSpeeds").setPosition(80, 40)
				.setSize(50, 20).setMode(ControlP5.SWITCH);
		
		// Set controls to window object
		((Toggle) bgColor).moveTo(window);
		((Knob) blurKnob).moveTo(window);
		((Knob) speedKnob).moveTo(window);
		((Toggle) rSpeeds).moveTo(window);
	}
	
	// ==================================================
	// Super Fast Blur v1.1, reworked by Michael Kontopoulos
	// by Mario Klingemann
	// http://incubator.quasimondo.com/processing/superfast_blur.php
	// ==================================================
	public void fastBlur(int[] img, int radius) {

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