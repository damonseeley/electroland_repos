package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.*;
import net.electroland.utils.ElectrolandProperties;

import org.apache.log4j.Logger;

import controlP5.ControlEvent;

public class EastBlurTest extends GothamPApplet {

	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	private int nStripes; // Num Stripes that begin on screen.
	public static float defaultScaler;

	public static float scalerAmt;
	public float blurAmt;
	public boolean blackOrWhite;
	public static boolean randomSpeeds;
	private int selector = 0; // Which color swatch from the props file to use.

	StripeGUIManager gui;
	ArrayList<Stripe> stripes;
	private float spawnRate;
	private long startTime = 0;
	private float percentComplete;

	private ElectrolandProperties props = GothamConductor.props;

	public static int[] stripeColors;
	ColorPalette cp;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		rectMode(CENTER);

		stripes = new ArrayList<Stripe>();
		// Populate the screen with several existing stripes.
		nStripes = props.getOptionalInt("wall", "East", "initialStripes");
		defaultScaler = (float) props.getOptionalInt("wall", "East",
				"initialScaler");

		cp = new ColorPalette(this); // Instantiate Color Palette by sampling
										// the listed swatch.
		stripeColors = cp.getPalette(selector);

		gui = new StripeGUIManager(this);

		for (int i = nStripes; i >= 0; i--)
			stripes.add(new Stripe(this, syncArea, i));
		// How often to generate a new stripe
		spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
		startTime = millis();

		logger.info("Initial OnScreen Stripes: " + nStripes);
		logger.info("Initial Speed Scaler: " + scalerAmt);
	}

	@Override
	public void drawELUContent() {

		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));

		// Handle Stripes
		for (int i = stripes.size() - 1; i >= 0; i--) {
			Stripe s = stripes.get(i);
			s.run();
			if (s.isOffScreen())
				stripes.remove(i);
		}

		// Timing Control for each new Stripe
		float inc = ((millis() - startTime) / (spawnRate)) * scalerAmt;
		percentComplete += inc;
		startTime = millis();
		if (percentComplete > 0.98) {
			stripes.add(new Stripe(this, syncArea));
			spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
			percentComplete = 0;
		}

		// Blur. Right now, blur is controlled by the vertical mouse component.
		loadPixels();
		FastBlur.performBlur(pixels, width, height, floor(blurAmt));
		updatePixels();
	}

	public void controlEvent(ControlEvent theEvent) {
		if (theEvent.isGroup() && theEvent.getName() == "whichSwatch") {
			selector = (int) theEvent.getValue();
			stripeColors = cp.getPalette(selector);
			logger.info("Switching to Swatch "
					+ (int) (theEvent.getValue() + 1) + " ("
					+ ColorPalette.getNumColors() + " Colors)");
		}
	}
}