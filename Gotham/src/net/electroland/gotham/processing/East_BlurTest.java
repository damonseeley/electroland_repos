package net.electroland.gotham.processing;

import net.electroland.gotham.processing.assets.Stripe;
import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.gotham.core.People;
import org.apache.log4j.Logger;

import de.looksgood.ani.Ani;
import processing.core.PImage;
import processing.core.PApplet;
import processing.core.PGraphics;

public class East_BlurTest extends GothamPApplet {

	@Override
	public void drawELUContent(){
	
	}
	/*private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(East.class);

	private Dimension syncArea;

	ArrayList<Stripe> stripes;
	int scaler = 5; // amt by which to scale down the offscreen texture
	private static long DURATION;
	private long startTime = 0;

	public PGraphics test = createGraphics(syncArea.width,syncArea.height, PApplet.P3D);
	@Override
	public void setup() {
		// syncArea is the area of the screen that will be synced to the lights.
		syncArea = this.getSyncArea();
		// our square's center will be the middle of the sync area.
		colorMode(HSB, 360, 100, 100);

//		Ani.init(this);
//		stripes = new ArrayList<Stripe>();
//		stripes.add(new Stripe(this));
		
		DURATION = 15000 / (long) Stripe.w;
	}

	@Override
	public void drawELUContent() {

		// get the presence grid (not doing anything with it yet.
		People pm = getPeople();
		if (pm != null) {
			logger.debug(pm);
		}

//		for (int i = stripes.size() - 1; i >= 0; i--) {
//			Stripe s = stripes.get(i);
//			s.run();
//			if (s.kill())
//				stripes.remove(i);
//		}

//		 if (System.currentTimeMillis() - startTime > DURATION){
//		 stripes.add(new Stripe(this));
//		 startTime = System.currentTimeMillis();
//		 }

		// off.loadPixels();
		// fastBlur(off, 4);
		// off.updatePixels();

		//image(Stripe.getOffScreenCanvas(), 0, 0, syncArea.width, syncArea.height);
		// Testing...
		test.beginDraw();
		test.fill(0, 100, 100);
		test.ellipse(width / 2, height / 2, 50, 50);
		test.endDraw();
		image(test, 0,0,syncArea.width, syncArea.height);
	}

	// ==================================================
	// Super Fast Blur v1.1, reworked by Michael Kontopoulos
	// by Mario Klingemann
	// http://incubator.quasimondo.com/processing/superfast_blur.php
	// ==================================================
	void fastBlur(PImage img, int radius) {

		if (radius < 1) {
			return;
		}
		int w = img.width;
		int h = img.height;
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
				p = img.pixels[yi + min(wm, max(i, 0))];
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
				p1 = img.pixels[yw + vmin[x]];
				p2 = img.pixels[yw + vmax[x]];

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
				img.pixels[yi] = 0xff000000 | (dv[rsum] << 16)
						| (dv[gsum] << 8) | dv[bsum];
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
	*/
}