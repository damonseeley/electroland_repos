package net.electroland.noho.graphics.generators;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.noho.graphics.ImageGenerator;

/**
 * Loads image sequence and renders animation.  Images don't have to be same size and mainframe (or even the same size as eachother).  Alpha is preserved
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class ImageFileAnimation extends ImageGenerator {
	Vector<BufferedImage> images = new Vector<BufferedImage>();
	
	String prefix;
	int zeroPrefix;
	String fileSuffix;
	int startNum; 
	int curImage = -1;
	int loop = 1;
	int origLoop;
	
	int offsetX =0;
	int offsetY =0;
	
	long timeOfNextFrame = -1;
	
	boolean isReady = false;

	
	long timePerImage;
	boolean isDone = false;
	public ImageFileAnimation(int width, int height) {
		super(width, height);
	}

	/**
	 * sets the location of the upper right hand cornder of image
	 * @param x
	 * @param y
	 */
	public void setPosition(int x, int y) {
		offsetX =x;
		offsetY =y;
	}

	
	/**
	 * loads images is a separate thread (needed to avoid rendering problems).  check isReady() to determine when loading is complete
	 * @param prefix - directory and filename
	 * @param startNum - first number in the sequence (usually 0 or 1)
	 * @param digitCnt - number of digits (eg "img001.png" should have digitCnt of 3)
	 * @param fileSuffix - everything aftet the number (eg ".png"
	 * @param fps - framerate for animation. Anything higher than programs framerate will be renders at programe rate (ie frames are not dropped)
	 */
	public void loadImagesAsync(String prefix, int startNum, int digitCnt, String fileSuffix, double fps) {
		this.prefix = prefix;
		this.zeroPrefix = digitCnt;
		this.fileSuffix = fileSuffix;
		this.startNum = startNum;
		setFPS(fps);
		new ImageLoaderThread().start();
	}


	public void loadImagesSync(String prefix, int startNum,int zeroPrefix, String fileSuffix, double fps) {
		this.prefix = prefix;
		this.zeroPrefix = zeroPrefix;
		this.fileSuffix = fileSuffix;
		this.startNum = startNum;
		setFPS(fps);
		loadImages();
	}
	public void reset() {
		loop = origLoop;
		curImage = -1;
		timeOfNextFrame = -1;
		isDone = false;
	}
	/**
	 * 
	 * @param i - number of times the sequence should be looped (1 results in a single play).  Use -1 to loop forever
	 */
	public void playCnt(int i) {
		origLoop = i;
		this.loop = i;
	}
	
	@Override
	/**
	 * @return true when done with loop else false
	 */
	public boolean isDone() {
		return isDone ;
	}

	/**
	 * 
	 * @return true when loadImagesAsync is finished loading images
	 */
	public boolean isReady() {
		return isReady;
	}
	@Override
	protected void render(long dt, long curTime) {
		if(isReady) {
			if(! isDone) {
				if(curTime >= timeOfNextFrame) {
					if(timeOfNextFrame <= 0) { // init
						timeOfNextFrame = curTime;
					}
					curImage++;
					timeOfNextFrame += timePerImage; // adjust for overplay
					if(curImage >= images.size()) {
						if(loop != 0) {
							loop--;
							curImage = 0;
						} else {
							curImage = images.size() -1;
							isDone = true;
						}
						
					}
					
				}
				Graphics2D g2d = image.createGraphics();
				clearBackground(g2d);
				g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
				g2d.drawImage(images.get(curImage), offsetX,offsetY, null);
			}
		}
		
	}
	
	
	public void setFPS(double fps) {
		timePerImage = (long) (1000.0 / fps);
	}
	
	protected void loadImages() {
		int curFile = startNum;
		boolean isOk = true;

		while(isOk) {
			try {
				images.add(ImageIO.read(new File(constructFilename(curFile++))));
			} catch (IOException e) {
				if(curFile == startNum + 1) {
					System.err.println("Unable to load animation file " + constructFilename(startNum) + "\n" + e);
				}
				isOk = false;
			}
		}
		isReady = true;
	}
	
	public String constructFilename(int i) {
		String numStr = Integer.toString(i);
		while(numStr.length() < zeroPrefix) {
			numStr = "0" + numStr;
		}
		return prefix+numStr+fileSuffix;

		
	}
	public class ImageLoaderThread extends Thread {
		public void run() {
			loadImages();
		}
	}
	
	



}
