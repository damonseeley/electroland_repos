package net.electroland.noho.graphics.generators;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import net.electroland.noho.graphics.ImageGenerator;
import net.electroland.noho.util.DeltaTimeInterpolator;

/**
 * Mans a single image (loaded from file) across from startPont to endPoint.  Alpha is preserved
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class ImagePanner extends ImageGenerator {

	boolean isDone = false;
	boolean isReady = false;
	boolean isOk = true;
	
	BufferedImage img;
	String filename;
	
	int x;
	int y;
	int dx;
	int dy;
	
	DeltaTimeInterpolator interp = new DeltaTimeInterpolator(0);

	
	public void reset() {
		isDone = false;
		interp.reset();
	}

	public ImagePanner(int width, int height) {
		super(width, height);
		
	}
	
	public void pan(int startX, int startY, int endX, int endY, long time) {
		x = startX;
		y = startY;
		dx = endX - startX;
		dy = endY - startY;
		interp.reset(time);
	}

	public void loadImageAsync(String filename) {
		this.filename = filename;

		new ImageLoaderThread().start();
	}

	
	protected void loadImage() {
			try {
				img = ImageIO.read(new File(filename));
				isReady = true;
			} catch (IOException e) {
					System.err.println("ImagePanner Unable to load animation file " + filename + "\n" + e);
				isOk = false;
			}
	}
	

	public class ImageLoaderThread extends Thread {
		public void run() {
			loadImage();
		}
	}


	@Override
	public boolean isDone() {
		return interp.isDone() || (! isOk);
	}

	public boolean isReady() {
		return isReady;
	}

	@Override
	protected void render(long dt, long curTime) {
		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		double d = 1.0-interp.interp(dt);
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		if(interp.isDone()) {
			g2d.drawImage(img,  (int) (x+ dx), (int) (y+dy), null);		
		} else {			
			g2d.drawImage(img, (int) (x+(dx * d)), (int) (y+(dy * d)), null);		
		}
	}


}
