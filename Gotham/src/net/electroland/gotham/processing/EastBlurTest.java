package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.Stripe;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.gotham.processing.assets.ColorPalette;
import net.electroland.utils.ElectrolandProperties;

import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;

import org.apache.log4j.Logger;

public class EastBlurTest extends GothamPApplet {

	public static boolean randomSpeeds;
	public static float scaler; // A scaler value to upsample or downsample the
								// duration of the tween
	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	private int nStripes; // Num Stripes that begin on screen.

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
	public static int selector = 0; //Which color swatch from the props file to use.
	private float spawnRate;
	private long startTime = 0;
	public float percentComplete;

	private ElectrolandProperties props = GothamConductor.props;
	public static int[] stripeColors;
	ColorPalette cp;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		rectMode(CENTER);
		
		cp = new ColorPalette(this); //Instantiate Color Palette by sampling the listed swatch.
		stripeColors = cp.getPalette();

		stripes = new ArrayList<Stripe>();
		// Populate the screen with several existing stripes.
		nStripes = props.getOptionalInt("wall", "East", "initialStripes");
		scaler = (float) props.getOptionalInt("wall", "East", "initialScaler");

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

		scaler = scalerAmt; // Point the class' scaler val to the knob

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
		FastBlur.performBlur(pixels, width, height, floor(blurAmt));
		updatePixels();
	}

	private void initGui() {
		control = new ControlP5(this);

		// Init window
		window = control
				.addControlWindow("Stripe_Control_Window", 100, 100, 200, 200)
				.hideCoordinates().setBackground(color(90));
		// Speed Scaler Knob
		speedKnob = control.addKnob("scalerAmt").setRange(0.2f, 3.5f)
				.setValue(1).setPosition(10, 100).setRadius(30)
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

}