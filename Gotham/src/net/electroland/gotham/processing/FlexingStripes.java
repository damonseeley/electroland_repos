package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.HashMap;

import net.electroland.gotham.core.GothamConductor;
//import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.*;
import net.electroland.utils.ElectrolandProperties;
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
	private int selector = 3; // Which color swatch from the props file to use.
	private float pScalerAmt;
	private boolean switchDirection;
	private boolean pinMode;
	private boolean insertMode;
	public static boolean flexMode;

	StripeGUIManager gui;
	ArrayList<Stripe> stripes;
	HashMap<String, String> affecters;

	private final boolean DEBUG = false;
	ColorPalette cp;
	private float wind;
	public static boolean widthShift;
	private ElectrolandProperties props = GothamConductor.props;
	private boolean insertStripe = false;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		pms = new PersonMouseSimulator(this, syncArea);

		stripes = new ArrayList<Stripe>();
		affecters = new HashMap<String, String>();

		cp = new ColorPalette(this);
		cp.createNewPalette(selector);
		gui = new StripeGUIManager(this);
		stripeTimer = new Timer(interval);
		paletteTimer = new Timer(10000);

		stripeTimer.start();
		paletteTimer.start();

		for (int i = 0; i < 17; i++) {
			stripes.add(new StripeFlexer(this, syncArea));
			((StripeFlexer) stripes.get(i)).forcePosition(syncArea.width
					- (i * 100));
		}
		wind = (float) (props
				.getOptionalDouble("wall", "East", "initialScaler") * 1.0f) / 3.5f;

		// I recommend turning off insert to test the affectors.
		// Pinning hasn't been updated for center-registerd stripes, so leave it
		// off.
		setFlexing(true);
		setPinning(false);
		setInsert(true);
		// affecters.put("SATURATION", "$150$40$100"); // radius, min, max
		// affecters.put("HUE", "$150"); // radius
		// affecters.put("WIDTH", "$50$1$2"); //radius, min scaler, max scaler
	}

	@Override
	public void drawELUContent() {

		// System.out.println(stripes.size());

		pms.update();
		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));
		noiseOffset = 0;// noise(millis()/100000.0f); //noramlized

		// Handle Stripes
		for (Stripe s : stripes) {
			s.update();
			if (stripes.indexOf(s) != 0) {
				s.setWidth(stripes.get(stripes.indexOf(s) - 1));
			}
			if (affecters.containsKey("SATURATION")) {
				s.performSaturationShift(pms, affecters.get("SATURATION"));
			}
			if (affecters.containsKey("HUE")) {
				s.performHueShift(pms, affecters.get("HUE"));
			}
			if (affecters.containsKey("WIDTH")) {
				((StripeFlexer) s).performWidthShift(pms,
						affecters.get("WIDTH"));

				widthShift = true;
			} else
				widthShift = false;

			if (pinMode)
				s.checkPinning(pms);

			if (!widthShift)
				s.display();

			// For testing. Prints the list index
			fill(360);
			textSize(25);
			text(stripes.indexOf(s), s.xpos, 50);
		}

		// This loops handles the shifting/offset of other stripes, based
		// on how many stripes (above loop) are being expanded by audience
		// presence in the With Affecter
		if (widthShift) {

			for (int i = 0; i < stripes.size(); i++) {
				Stripe s = stripes.get(i);
				float offset = 0;
				if (!s.isGrowing()) {
					for (int j = 0; j < stripes.size(); j++) {
						if (stripes.get(j).isGrowing()) {
							offset += (s.w / (j - i));
						}
					}
					s.setWidth((float) (1.0 - (stripes.size() / 50))); // not
																		// working
				}
				if (stripes.indexOf(s) != 0) {
					s.setWidth(stripes.get(stripes.indexOf(s) - 1));
				}
				s.display(offset);
			}

		}

		// removal of offscreen stripes.
		for (int i = stripes.size() - 1; i >= 0; i--) {
			Stripe s = stripes.get(i);
			if (s.isOffScreen())
				stripes.remove(i);
		}

		if (insertMode) {
			if (insertStripe) {
				int index = -1;
				float pos = 0;

				for (Stripe s : stripes) {
					if (s.containsLocation(pms.getZone())) {
						index = stripes.indexOf(s);
						//pos = the space in between index and index-1
						pos = stripes.get(index).getLeftSide()-10;
					}
				}
				if (index >= 0) {
					System.out.println("Inserting a Stripe at position "
							+ (index+1));

					stripes.add(index+1, new StripeFlexer(this, syncArea, pos));
					//stripes.get(index+1).setWidth(50);
				}
			}

			// When a new strip is gonna be added, pause all the prior stripes
			for (Stripe s : stripes) {
				if (s.stillEasing && millis() > 5000) {
					for (int i = stripes.size() - 1; i > stripes.indexOf(s); i--) {
						// float loc = s.getLocation();
						// s.forcePosition(loc - (s.w * 2));
						stripes.get(i).getBehavior().pause();
					}
				}
			}
			for (Stripe s : stripes) {
				if (s.justFinishedEasing() && millis() > 5000) {
					System.out.println("resume");
					for (int i = stripes.size() - 1; i > stripes.indexOf(s); i--) {
						stripes.get(i).getBehavior().resume();
					}
				}
			}
		}

		if ((Stripe.scalerAmt >= 0 && pScalerAmt < 0)
				|| (Stripe.scalerAmt <= 0 && pScalerAmt > 0)) {
			switchDirection = true;
		}

		if (switchDirection) {
			if (DEBUG)
				System.out.println("Direction Change******************");
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
				System.out.print(s.getBehavior().toString() + " " + s.xpos
						+ "\t");
			}
			System.out.println();
		}

		// Timing Controls for each new Stripe
		if (stripeTimer.isFinished()) {
			Stripe lastOne = stripes.get(stripes.size() - 1);
			
			if (lastOne.getLeftSide() > -MoveBehavior.offset) {
				stripes.add(new StripeFlexer(this, syncArea));

				if(flexMode)
					stripeTimer.reset((long) (1000.0 + (Math.random() *spawnScaler)));
				else
					stripeTimer.reset((long) spawnScaler); //Around 1900s for 100px wide stripes

			}
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
		insertStripe = false;
	}

	@Override
	public void mousePressed() {
		if (insertMode)
			insertStripe = true;
	}

	public void setPinning(boolean p) {
		pinMode = p;
	}
	public void setFlexing(boolean f){
		flexMode = f;
	}

	public void setInsert(boolean i) {
		insertMode = i;
	}

	// Processing Key Event shortcut. Used just to test wind.
	// One left/right arrow key represents a person passing by.
	// Assuming a "wind" value between -1 and 1
	// *** Note: Key events aren't working. Did we disable key events someplace?
	@Override
	public void keyPressed() {
		System.out.println("key");
		if (key == CODED) {
			if (keyCode == RIGHT) {
				wind += 0.1;
			} else if (keyCode == LEFT) {
				wind -= 0.1;
			}
		}

		wind = constrain(wind, -1f, 1f);
		Stripe.setWindScaler(wind);
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
			Stripe.setScalerAmtFromKnob(theEvent.getController().getValue());
			if (!DEBUG) {
				logger.info("Resetting Speed Scaler To: "
						+ theEvent.getController().getValue());
			}
		} else if (theEvent.getController().getName() == "rScaler") {
			Stripe.setRandomScaler(theEvent.getController().getValue());
			logger.info("Resetting Stripe Randomness To: "
					+ theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "spawnScaler") {
			logger.info("Resetting Spawn Rate To: "
					+ theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "randomSpeeds") {
			if (!widthShift)
				Stripe.setUseRandomSpeeds(theEvent.getController().getValue());
			else
				Stripe.setUseRandomSpeeds(-1);
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