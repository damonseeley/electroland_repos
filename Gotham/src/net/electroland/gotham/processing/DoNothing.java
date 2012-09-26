package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.Stripe;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.utils.ElectrolandProperties;

import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Toggle;

import org.apache.log4j.Logger;

public class DoNothing extends GothamPApplet {

	public static boolean randomSpeeds;
	public static float scaler; // A scaler value to upsample or downsample the duration of the tween	
	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;

	public float scalerAmt;
	public float blurAmt;
	public boolean blackOrWhite;

	ArrayList<Stripe> stripes;
	public float percentComplete;

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		
	}

	@Override
	public void drawELUContent() {
		background(color(0, 0, 20));

		loadPixels();
		updatePixels();
	}
}