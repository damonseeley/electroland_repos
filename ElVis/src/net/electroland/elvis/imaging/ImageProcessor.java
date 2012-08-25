package net.electroland.elvis.imaging;

import java.awt.image.BufferedImage;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicReference;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public abstract class ImageProcessor extends Thread implements ImageReceiver {
	protected boolean isRunning = true;

	LinkedBlockingQueue<IplImage> queue = new LinkedBlockingQueue<IplImage>();
	Vector<IplImage> toProcess = new Vector<IplImage>();

	int frameCnt = 0;

	long startTime;
	
	public int warningFrameSize = 15;
	

	AtomicReference<IplImage> curImage = new AtomicReference<IplImage>();

	public ImageProcessor() {
	}

	
	public void addImage(BufferedImage i) {
		queue.add(IplImage.createFrom(i));
	}
	public void addImage(IplImage i) {
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

		IplImage result = null;
		while(isRunning) {
			
			try {
				IplImage img = queue.take();
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

	public IplImage getImageIplImage() {
		return curImage.get();
	}

	public BufferedImage getBufferedImage() {
		IplImage img = curImage.get();
		if(img != null) return img.getBufferedImage();
		return null;
	}
	
	public float getFPS() {
		return (1000.0f * frameCnt) / ((float) (System.currentTimeMillis() - startTime));
	}

	public void receiveErrorMsg(Exception cameraException) {
		System.out.println(cameraException);
	}
	
	public abstract IplImage process(IplImage img);
}
