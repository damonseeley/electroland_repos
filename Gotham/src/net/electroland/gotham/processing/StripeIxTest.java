package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.assets.ColorPalette;
import net.electroland.gotham.processing.assets.StripeSimple;
import net.electroland.gotham.processing.assets.Timer;

import org.apache.log4j.Logger;

public class StripeIxTest extends GothamPApplet {

	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	//private int nStripes; // Num Stripes that begin on screen.

	private long interval = 3000;
	private Timer timer;

	ArrayList<StripeSimple> stripes;

	ColorPalette cp;

	@Override
	public void handle(GridData t) {
		//System.out.println("--");
		//for(int y =0; y < t.height; y++) {
			//for(int x =0; x < t.width; x++) {

				//System.out.print(t.getValue(x, y) + "  ");
			//}
			//System.out.println("");
		//}
	}

	@Override
	public void setup() {
		System.out.println("hello");
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);

		stripes = new ArrayList<StripeSimple>();

		//nStripes = props.getOptionalInt("wall", "East", "initialStripes");

		cp = new ColorPalette(this);
		cp.createNewPalette(0);
		timer = new Timer(interval);
		timer.start();

		stripes.add(new StripeSimple(this, syncArea, (int)Math.random()*this.syncArea.width));
	}

	@Override
	public void drawELUContent() {

		background(color(0, 0, 0));

		// Handle Stripes
		for (int i = stripes.size() - 1; i >= 0; i--) {
			StripeSimple s = stripes.get(i);
			s.run();
			if(i!=0)

				if (s.isOffScreen())
					stripes.remove(i);
		}

		// Timing Controls for each new Stripe
		if (timer.isFinished()) {
			int xloc = (int)(Math.random()*this.syncArea.width);
			logger.info("creating new stripe at: " + xloc + " where syncarea.width = " + this.syncArea.width);
			stripes.add(new StripeSimple(this, syncArea, xloc));
			timer.reset((long)(1000.0 + (Math.random()*5))); //Randomize the timer each time. Result: stripes of diff widths
		}


		loadPixels();
		updatePixels();
	}


}