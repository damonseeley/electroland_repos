package net.electroland.gotham.processing;

import java.util.ArrayList;

import net.electroland.gotham.processing.assets.Stripe;

import org.apache.log4j.Logger;

public class DoNothing extends GothamPApplet {

	public static boolean randomSpeeds;
	public static float scaler; // A scaler value to upsample or downsample the duration of the tween	
	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);

	public float scalerAmt;
	public float blurAmt;
	public boolean blackOrWhite;

	ArrayList<Stripe> stripes;
	public float percentComplete;

	@Override
	public void setup() {
		
	}

	@Override
	public void drawELUContent() {
		background(color(40, 40, 40));

		loadPixels();
		updatePixels();
	}
}