package net.electroland.elvis.imaging;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicReference;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

public abstract class ImageProcessor extends Thread implements ImageReceiver {
	protected boolean isRunning = true;

	LinkedBlockingQueue<BufferedImage> queue = new LinkedBlockingQueue<BufferedImage>();
	Vector<BufferedImage> toProcess = new Vector<BufferedImage>();

	int frameCnt = 0;

	long startTime;
	
	public int warningFrameSize = 15;
	
	public int w;
	public int h;

	AtomicReference<BufferedImage> curImage = new AtomicReference<BufferedImage>();

	public ImageProcessor(int w, int h) {
		this.w = w;
		this.h = h;
	}

	
	public void addImage(BufferedImage i) {
		queue.add(i);
	}


	public void resetFPSCalc() {
		frameCnt = 0;
		startTime = System.currentTimeMillis();
	}
	
	public void start() {
		isRunning = true;
		queue.drainTo(toProcess);
		toProcess.clear();
		super.start();
	}
	
	public void stopRunning() {
		isRunning = false;
	}


	public void run() {
		startTime = System.currentTimeMillis();

		BufferedImage result = null;
		while(isRunning) {
			
			try {
				BufferedImage img = queue.take();
				if(! queue.isEmpty()) { // if behind catch up
					queue.drainTo(toProcess);
					if(toProcess.size() > warningFrameSize) {
						System.err.println("warning: ImageProcessor is thowing away " + toProcess.size() + " frames");
					}
					frameCnt++;
					result = process(toProcess.get(0));					
					int i = 2;
					while(i < toProcess.size()) {						
						// lets do a exponential drop off to smooth droped frames
						frameCnt++;
						result = process(toProcess.get(i));											
						i *= 2;
					}
					toProcess.clear();
				} else {
					frameCnt++;
					result = process(img);					
				}
				if(curImage != null) {
					curImage.set(result);
				}


			} catch (InterruptedException e) {
			}

		}
	}

	public RenderedImage getImage() {
		return curImage.get();
	}

	
	public float getFPS() {
		return (1000.0f * frameCnt) / ((float) (System.currentTimeMillis() - startTime));
	}

	public abstract BufferedImage process(BufferedImage img);
}
