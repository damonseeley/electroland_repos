package net.electroland.gotham.processing;

import java.awt.Dimension;
import org.apache.log4j.Logger;

public class StripeTestPApplet extends GothamPApplet {

	private static final long serialVersionUID = 449793686955037868L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	
	public int numStripes = 6;
	public float inc;
	public int[] hues = new int[numStripes];
	private final int offset = 20; //A little bleed to account for the black stripes between faded stripes.
	
	//Select Horizontal or Vertical stripes
	private boolean horizontal = true;
	
	private boolean useMouse = true; //If you set this to false, dial in your blur const below
	private final int BLUR_AMT = 20;
	private final int BLUR_MAX = 100; //If using the mouse, this constrains the blurriness. 

	@Override
	public void setup() {
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);
		inc = horizontal ? syncArea.width / numStripes : syncArea.height/numStripes;
		for (int i = 0; i < hues.length; i++)
			hues[i] = color((float) Math.random() * 360, 90, 90);
	}

	@Override
	public void drawELUContent() {
		background(0);
		for (int i = 0; i < numStripes; i++) {
			fill(hues[i]);
			if(horizontal)
				rect(i * inc-offset, 0, inc+offset*2, syncArea.height*2);
			else
				rect(0, i*inc-offset, syncArea.width, inc+offset*2);
		}
		
		int blurAmt = useMouse ? floor(map(mouseX,0,width,0,BLUR_MAX)) : BLUR_AMT;
		
		loadPixels();
		fastBlur(pixels, blurAmt);
		updatePixels();
	}

	// ==================================================
	// Super Fast Blur v1.1, reworked by Michael Kontopoulos
	// originally by Mario Klingemann
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
				img[yi] = 0xff000000 | (dv[rsum] << 16)
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

}
