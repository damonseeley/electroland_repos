package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
//import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.*;
//import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;
import controlP5.ControlEvent;

public class FlexingStripes extends GothamPApplet {
	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;

	public static float spawnScaler;
	public float blurAmt;
	public boolean blackOrWhite;
	public PersonMouseSimulator pms;

	private long interval = 3000;
	private Timer stripeTimer;
	private Timer paletteTimer;
	public static float noiseOffset;
	private int selector = 5; // Which color swatch from the props file to use.
	private float pScalerAmt;
	private boolean switchDirection;

	StripeGUIManager gui;
	ArrayList<Stripe> stripes;

	private final boolean DEBUG = true;
	ColorPalette cp;
	
	//A hook for adding in a stripe?
	//List of Affectors. 

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		pms = new PersonMouseSimulator(this, syncArea);

		stripes = new ArrayList<Stripe>();

		cp = new ColorPalette(this);
		cp.createNewPalette(selector);
		gui = new StripeGUIManager(this);
		stripeTimer = new Timer(interval);
		paletteTimer = new Timer(10000);

		stripeTimer.start();
		paletteTimer.start();

		for (int i=0; i<7; i++) {
			stripes.add(new StripeFlexRight(this, syncArea));
			((StripeFlexRight) stripes.get(i)).forcePosition(syncArea.width - (i*200));
		}

	}

	@Override
	public void drawELUContent() {
		pms.update();

		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));
		noiseOffset = 0;// noise(millis()/100000.0f); //noramlized

		// Handle Stripes
		for (int i = stripes.size() - 1; i >= 0; i--) {
			Stripe s = stripes.get(i);
			s.update();
			s.display();
			//check for affectors
			//apply existing affector to the current stripe.

			// if(pms.onScreen())
			// s.checkHover(pms.getLocation(), pms.standing());
			s.checkHover(pms);

			if (i != 0)
				s.setWidth(stripes.get(i - 1)); // Set the width of this stripe,
												// based on the guy in front.

			if (s.isOffScreen())
				stripes.remove(i);
		}
		
		if ((Stripe.scalerAmt >= 0 && pScalerAmt < 0)
				|| (Stripe.scalerAmt <= 0 && pScalerAmt > 0)) {
			switchDirection = true;
		}


		if (switchDirection) {
			if(DEBUG) System.out.println("Direction Change******************");
			for (Stripe s : stripes) {
				float tx = s.getLocation();
				if (Stripe.scalerAmt < 0) {
					s.setBehavior(new MoveLeft(this, syncArea, tx));
				} else {
					s.setBehavior(new MoveRight(this, syncArea, tx));
				}
			}

		}

		if (DEBUG) {
			for (Stripe s : stripes) {
				System.out.print(s.getBehavior().toString() + "\t");
			}
			System.out.println();
		}

		// Timing Controls for each new Stripe
		if (stripeTimer.isFinished()) {
			if (Stripe.scalerAmt > 0)
				stripes.add(new StripeFlexRight(this, syncArea));
			else if (Stripe.scalerAmt < 0)
				stripes.add(new StripeFlexLeft(this, syncArea));

			stripeTimer.reset((long) (1000.0 + (Math.random() * spawnScaler)));
		}

		// Timer to pick new color palettes
		if (paletteTimer.isFinished()) {
			int n = (int) (Math.random() * ColorPalette.getNumSwatches());
			cp.createNewPalette(n);
			logger.info("Created new color palette " + n);
			paletteTimer.reset((long) (600000 + random(-300000, 300000)));
		}

		// Blur
		loadPixels();
		FastBlur.performBlur(pixels, width, height, floor(blurAmt));
		updatePixels();
		pScalerAmt = Stripe.scalerAmt;
		switchDirection = false;
	}

	// Event method for the GUI knobs
	public void controlEvent(ControlEvent theEvent) {

		if (theEvent.isGroup() && theEvent.getName() == "whichSwatch") {
			selector = (int) theEvent.getValue();
			cp.createNewPalette(selector);
			logger.info("Switching to Swatch "
					+ (int) (theEvent.getValue() + 1) + " ("
					+ ColorPalette.getNumColors() + " Colors)");
		} else if (theEvent.getController().getName() == "blurAmt") {
			logger.info("Resetting Blur Amount To: " + blurAmt);
		} else if (theEvent.getController().getName() == "scalerAmt") {
			Stripe.setScalerAmt(theEvent.getController().getValue());
			if(!DEBUG) {logger.info("Resetting Speed Scaler To: "
					+ theEvent.getController().getValue());}
		} else if (theEvent.getController().getName() == "rScaler") {
			Stripe.setRandomScaler(theEvent.getController().getValue());
			logger.info("Resetting Stripe Randomness To: "
					+ theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "spawnScaler") {
			logger.info("Resetting Spawn Rate To: "
					+ theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "randomSpeeds") {
			Stripe.setUseRandomSpeeds(theEvent.getController().getValue());
			logger.info("Randomize Speed? "
					+ theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "blackOrWhite") {
			logger.info("Black Background? " + blackOrWhite);
		} else if (theEvent.getController().getName() == "grow") {
			Stripe.setGrow(theEvent.getController().getValue());
			logger.info("Growing Stripes? "
					+ theEvent.getController().getValue());
		}

	}
}