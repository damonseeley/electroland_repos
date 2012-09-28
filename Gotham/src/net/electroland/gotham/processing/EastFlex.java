package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
//import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.*;
//import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;
import controlP5.ControlEvent;

public class EastFlex extends GothamPApplet {

	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	public static float spawnScaler;
	public float blurAmt;
	public boolean blackOrWhite;
	
	private long interval = 3000;
	private Timer stripeTimer;
	private Timer paletteTimer;
	public static float noiseOffset;
	private int selector = 0; // Which color swatch from the props file to use.

	StripeGUIManager gui;
	ArrayList<StripeFlex> stripes;

	//private ElectrolandProperties props = GothamConductor.props;

	ColorPalette cp;

	@Override
	public void setup() {
		System.out.println("hello");
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);

		stripes = new ArrayList<StripeFlex>();

		cp = new ColorPalette(this);
		cp.createNewPalette(0);
		gui = new StripeGUIManager(this);
		stripeTimer = new Timer(interval);
		paletteTimer = new Timer(10000);
		stripeTimer.start();
		paletteTimer.start();

		stripes.add(new StripeFlex(this, syncArea));
		stripes.get(0).forcePosition(syncArea.width);
		stripes.add(new StripeFlex(this, syncArea));
	}

	@Override
	public void drawELUContent() {

		float bri = blackOrWhite ? 0 : 100;
		background(color(0, 0, bri));
		noiseOffset = noise(millis()/100000.0f); //noramlized

		// Handle Stripes
		for (int i = stripes.size() - 1; i >= 0; i--) {
			StripeFlex s = stripes.get(i);
			s.run();
			if(i!=0)
				s.setWidth(stripes.get(i-1)); //Set the width of this stripe, based on the guy in front.
			
			if (s.isOffScreen())
				stripes.remove(i);
		}

		// Timing Controls for each new Stripe
		if (stripeTimer.isFinished()) {
			stripes.add(new StripeFlex(this, syncArea));
			stripeTimer.reset((long)(1000.0 + (Math.random()*spawnScaler))); //Randomize the timer each time. Result: stripes of diff widths
		}
		if(paletteTimer.isFinished()){
			int n = (int) (Math.random()*ColorPalette.getNumSwatches());
			cp.createNewPalette(n);
			System.out.println("Created new color palette " + n);
			paletteTimer.reset((long)(600000 + random(-300000, 300000))); //random interval between 5 and 15 mins
		}
	
		//System.out.println(stripes.size());

		loadPixels();
		FastBlur.performBlur(pixels, width, height, floor(blurAmt));
		updatePixels();
	}

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
			StripeFlex.setScalerAmt(theEvent.getController().getValue());
			logger.info("Resetting Speed Scaler To: " + theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "rScaler") {
			StripeFlex.setRandomScaler(theEvent.getController().getValue());
			logger.info("Resetting Stripe Randomness To: " + theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "spawnScaler") {
			logger.info("Resetting Spawn Rate To: " + theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "randomSpeeds") {
			StripeFlex.setUseRandomSpeeds(theEvent.getController().getValue());
			logger.info("Randomize Speed? " + theEvent.getController().getValue());
		} else if (theEvent.getController().getName() == "blackOrWhite") {
			logger.info("Black Background? " + blackOrWhite);
		}

	}
}