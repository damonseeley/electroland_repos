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
	private ElectrolandProperties props = GothamConductor.props;

	public static float spawnScaler; //GUI fields
	public float blurAmt;
	public boolean blackOrWhite;
	public PersonMouseSimulator pms;

	private long interval = 3000; //Timing Fields
	private Timer stripeTimer;
	// private Timer paletteTimer;
	public static float noiseOffset;
	private float pScalerAmt;
	private boolean switchDirection;
	private boolean pinMode;
	public static boolean insertMode;
	public static boolean flexMode;
	public static boolean widthShift;
	private boolean insertStripe = false;
	public Controller controller;
	ColorPalette cp;
	public int triggeredZone = -1;

	public StripeGUIManager gui;
	public static ArrayList<Stripe> stripes;
	public HashMap<String, String> affecters;
	private final boolean DEBUG = false;
	private float wind;
	public static int[] accentColors;
	
	/*	Below:
	 *  Selector = Which color palette to begin with.
	 *  numZones = How many zones from 0 to num-1
	 */
	private int selector = 0; 
	public static final int numZones = 5; 
	

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
		// paletteTimer = new Timer(10000);

		stripeTimer.start();
		// paletteTimer.start();

		
		/*
		 * Demo 1: Stripe Insertion with a desaturated bg and a "hot" colored new stripe
		 * Flexing, Pinning: False, Random Speeds: -1, Insert: True, All Affecters out except saturation if you want.
		 * Demo 2: Hue and Saturation Affecters
		 * Flexing, Inserting, Pinning: False, Random Speeds; -1, Width Affecter commented out
		 * Demo 3: Width Affecter
		 * Everything False, Random Speeds -1, Hue/Sat Affecters commented out.
		 * Demo 4: Autonomous Stripes
		 * Flexing: True, Random Speeds: 1,  Everything else False, Every affecter commented out.
		 */
	
		setController(Controller.MOUSE); // Options: MOUSE, HOOK

		setFlexing(false);
		Stripe.setUseRandomSpeeds(-1); //Set to 1 to force random speeds if you want flexing.
		setInsert(false);
		setPinning(false);
		affecters.put("SATURATION", "$80$15$100"); // radius, min, max
		affecters.put("WIDTH", "$50$1$2"); // radius, min scaler, max scaler
		//affecters.put("HUE", "$80"); // radius
	
		/*
		 * "Hot" colors for hue affecter and stripe insertion.
		 */
		accentColors = new int[3];
		accentColors[0] = color(30,100,100); //in hsv
		accentColors[1] = color(317, 100,100);
		accentColors[2] = color(0, 100,100);
		
		
		for (int i = 0; i <17; i++) {	
			stripes.add(new StripeFlexer(this, syncArea));
			((StripeFlexer) stripes.get(i)).forcePosition(syncArea.width- (i * 100));
		}
		wind = (float) (props
				.getOptionalDouble("wall", "East", "initialScaler") * 1.0f) / 3.5f;
	}

	@Override
	public void drawELUContent() {

		pms.update();
		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));
		noiseOffset = 0;// noise(millis()/100000.0f); //noramlized
		
		// Handle Stripes
		for (Stripe s : stripes) {
			s.update();
			
			/*
			 //Original Attempt at bi-directional flexing
			if(stripes.indexOf(s)!=0 
					&& stripes.get(stripes.indexOf(s)-1 ).boundaryStripe){
				s.setWidth(stripes.get(0));
			}
			else if(s.relic){
				if(!s.boundaryStripe)//(stripes.indexOf(s) != stripes.size()-1)
					s.setWidth(stripes.get(stripes.indexOf(s)+1));
			}
			else{
				if (stripes.indexOf(s) != 0 ) 
					s.setWidth(stripes.get(stripes.indexOf(s) - 1));
			}*/
			
			//Hacky comprimise. Any "leftover" stripes oncreen after a dir change, don't flex.
			if(stripes.indexOf(s) != 0){
				if(!s.leftover && !stripes.get(stripes.indexOf(s)-1).leftover)
					s.setWidth(stripes.get(stripes.indexOf(s) - 1));
				else if(!s.leftover && stripes.get(stripes.indexOf(s)-1).leftover){
					s.setWidth(stripes.get(0));
				} 
			}


			// Check Saturation Affecter
			if (affecters.containsKey("SATURATION")) {
				switch (controller) {
				case MOUSE:
					if (pms.getLocation().getY() >= syncArea.height - 50) {
						s.performSaturationShift(
								zoneToCoord(floor(map(mouseX, 0,
										syncArea.width, 0, numZones))),
								affecters.get("SATURATION"));
					} else
						s.performSaturationShift(-1,
								affecters.get("SATURATION"));
					break;
				case HOOK:
					if (triggeredZone > 0)
						s.performSaturationShift(zoneToCoord(triggeredZone),
								affecters.get("SATURATION"));
					else
						s.performSaturationShift(-1,
								affecters.get("SATURATION"));
					break;
				}
			}

			// Check Hue Affecter
			if (affecters.containsKey("HUE")) {
				switch (controller) {
				case MOUSE:
					if (pms.getLocation().getY() >= syncArea.height - 50) {
						s.performHueShift(
								zoneToCoord(floor(map(mouseX, 0,
										syncArea.width, 0, numZones))),
								affecters.get("HUE"));
					} else
						s.performHueShift(-1, affecters.get("HUE"));
					break;
				case HOOK:
					if (triggeredZone > 0)
						s.performHueShift(zoneToCoord(triggeredZone),
								affecters.get("HUE"));
					else
						s.performHueShift(-1, affecters.get("HUE"));
					break;
				}

			}
			// Check Width Affecter
			if (affecters.containsKey("WIDTH")) {
				widthShift = true;
				switch (controller) {
				case MOUSE:
					if (pms.getLocation().getY() >= syncArea.height - 50) {
						((StripeFlexer) s).performWidthShift(
								zoneToCoord(floor(map(mouseX, 0,
										syncArea.width, 0, numZones))),
								affecters.get("WIDTH"));
					} else
						((StripeFlexer) s).performWidthShift(-1,
								affecters.get("WIDTH"));
					break;
				case HOOK:
					if (triggeredZone > 0)
						((StripeFlexer) s).performWidthShift(
								zoneToCoord(triggeredZone),
								affecters.get("WIDTH"));
					else
						((StripeFlexer) s).performWidthShift(-1,
								affecters.get("WIDTH"));
					break;
				}

			} else {
				widthShift = false;
			}

			if (pinMode)
				s.checkPinning(pms);

			// Unless we're using widthShift, use this display method
			if (!widthShift)
				s.display();

			// For testing. Prints the list index
			//fill(360);
			//textSize(25);
			//text(stripes.indexOf(s), s.xpos, 50);
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

		//Stripe Insertion
		switch (controller) {
		case MOUSE:
			if (insertMode) {
				insertOneStripe(mouseX);
			}
			break;
		case HOOK:
			if (insertMode && triggeredZone > 0)
				insertStripe = true;
				insertOneStripe(zoneToCoord(triggeredZone));
			break;
		}

		// Switching Directions
		if ((Stripe.scalerAmt >= 0 && pScalerAmt < 0)
				|| (Stripe.scalerAmt <= 0 && pScalerAmt > 0)) {
			switchDirection = true;
			//stripes.get(0).boundaryStripe = true;
			stripes.get(stripes.size()-1).boundaryStripe = true;
			
			for(Stripe n : stripes)
				n.leftover = true;
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

		/*
		 *   Timing Controls for each new Stripe
		 */
		if (stripeTimer.isFinished()) {
			if(Stripe.scalerAmt >= 0){
				Stripe lastOne = stripes.get(stripes.size() - 1);
				if (lastOne.getLeftSide() > -MoveBehavior.offset
						&& lastOne.getBehavior().pauseState() == false) {
					stripes.add(new StripeFlexer(this, syncArea));

					if (flexMode)
						stripeTimer.reset((long) (1000.0 + (Math.random() * spawnScaler)));
					else
						stripeTimer.reset((long) spawnScaler);
				}
			} else {
				Stripe lastOne = stripes.get(stripes.size() - 1);
				if (lastOne.getRightSide() < syncArea.width+MoveBehavior.offset
						&& lastOne.getBehavior().pauseState() == false) {
					stripes.add(new StripeFlexer(this, syncArea));

					if (flexMode)
						stripeTimer.reset((long) (1000.0 + (Math.random() * spawnScaler)));
					else
						stripeTimer.reset((long) spawnScaler);
				}
			}
		}

		/*
		 * // Timer to pick new color palettes if (paletteTimer.isFinished()) {
		 * int n = (int) (Math.random() * ColorPalette.getNumSwatches());
		 * cp.createNewPalette(n); logger.info("Created new color palette " +
		 * n); paletteTimer.reset((long) (600000 + random(-300000, 300000))); }
		 */

		// Blur
		loadPixels();
		FastBlur.performBlur(pixels, width, height, floor(blurAmt));
		updatePixels();
		pScalerAmt = Stripe.scalerAmt;
		switchDirection = false;
		insertStripe = false;
	}

	public void zoneTriggered(int zone) {
		// Dummy code. We need some way to turn off the affectors when this
		// function isn't being called...
		// Like... you send -1 once after a person leaves a Zone?
		triggeredZone = zone;
	}

	// Mouse Event used to test Stripe Insertion
	@Override
	public void mousePressed() {
		if (insertMode)
			insertStripe = true;
		// The draw loop sets it back to false, so it's only active for 1 frame.
		// Hook from the CV class should take care of this...
	}

	public void setPinning(boolean p) {
		pinMode = p;
	}

	public void setFlexing(boolean f) {
		flexMode = f;
	}

	public void setInsert(boolean i) {
		insertMode = i;
	}

	public float zoneToCoord(int zone) {
		float inc = syncArea.width / (numZones * 2);
		return ((syncArea.width / numZones) * zone) + inc;
	}

	public void setController(Controller c) {
		this.controller = c;
	}

	public void insertOneStripe(float input) {
		if (insertStripe) {
			int index = -1;
			float pos = 0;

			for (Stripe s : stripes) {
				if (s.containsLocation(input)) {
					index = stripes.indexOf(s);
					// pos = the space in between index and index-1
					pos = stripes.get(index).getLeftSide() - 10;
				}
			}
			if (index >= 0) {
				System.out.println("Inserting a Stripe at position "
						+ (index + 1));

				stripes.add(index + 1, new StripeFlexer(this, syncArea, pos));
			}
		}

		// When a new strip is gonna be added, pause all the prior stripes
		for (int i = 0; i < stripes.size(); i++) {
			Stripe s = stripes.get(i);
			if (s.isNew && s.stillEasing && millis() > 5000) {
				for (int j = stripes.size() - 1; j > stripes.indexOf(s); j--) {
					Stripe ps = stripes.get(j);
					float loc = ps.getLocation();
					ps.forcePosition(loc - (ps.w / 4));
					ps.getBehavior().pause();
				}
			}
		}

		for (int i = 0; i < stripes.size(); i++) {
			Stripe s = stripes.get(i);
			if (s.isNew && s.justFinishedEasing() && millis() > 5000) {
				// System.out.println("resume");
				for (int j = stripes.size() - 1; j > stripes.indexOf(s); j--) {
					stripes.get(j).getBehavior().resume();
				}
			}
		}
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