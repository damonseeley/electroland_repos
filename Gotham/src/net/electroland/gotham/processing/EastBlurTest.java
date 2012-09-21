package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;

import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.Stripe;
import net.electroland.utils.ElectrolandProperties;

import org.apache.log4j.Logger;

public class EastBlurTest extends GothamPApplet {

	public static boolean randomSpeeds = true;

	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;

	ArrayList<Stripe> stripes;
	private float spawnRate;
	private long startTime = 0;
	
	private ElectrolandProperties props = GothamConductor.props;

	@Override
	public void setup() {
		// syncArea is the area of the screen that will be synced to the lights.
		syncArea = this.getSyncArea();
		// our square's center will be the middle of the sync area.
		colorMode(HSB, 360, 100, 100);
		rectMode(CENTER);

		stripes = new ArrayList<Stripe>();
		for(int i=6; i>=0; i--) stripes.add(new Stripe(this, syncArea, i));
		// How often to generate a new stripe
		spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
		startTime = millis();
		
		//PROPS FILE EXAMPLE
		logger.info("MICHAEL - here are some props from the global prop file");
		int numStripesTemp = props.getOptionalInt("wall", "East", "stripes");
		logger.info("Number of stripes = " + numStripesTemp);
	}

	@Override
	public void drawELUContent() {
		background(color(0,0,0));
		
		for (int i = stripes.size() - 1; i >= 0; i--) {
			Stripe s = stripes.get(i);
			s.run();
			if (s.isOffScreen())
				stripes.remove(i);
		}
		
		//System.out.println(stripes.size());
		
		if (millis() - startTime >= (spawnRate - 100)) {
			stripes.add(new Stripe(this, syncArea));
			spawnRate = stripes.get(stripes.size() - 1).getSpawnRate();
			startTime = millis();
		}
		
		// Right now, blur is controlled by the vertical mouse component.
		loadPixels();
		fastBlur(pixels, floor(map(mouseY, 0, height, 1, 50)));
		updatePixels();
	}

	// ==================================================
	// Super Fast Blur v1.1, reworked by Michael Kontopoulos
	// by Mario Klingemann
	// http://incubator.quasimondo.com/processing/superfast_blur.php
	// ==================================================
	void fastBlur(int[] img, int radius) {

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