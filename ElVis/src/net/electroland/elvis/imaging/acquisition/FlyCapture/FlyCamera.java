package net.electroland.elvis.imaging.acquisition.FlyCapture;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.FlyCaptureFrameGrabber;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.FrameGrabber.ImageMode;

public class FlyCamera extends Thread implements ImageAcquirer {
	int width;
	int height;

	FlyCaptureFrameGrabber frameGrabber;
	ImageReceiver imageReceiver;
	boolean isRunning;

	public FlyCamera (ImageReceiver imageReceiver, int dev, int w, int h) throws Exception {
		this.width = w;
		this.height = h;
		frameGrabber = new FlyCaptureFrameGrabber(dev);
//		frameGrabber.setImageHeight(height);
//		frameGrabber.setImageWidth(width);
		frameGrabber.setImageMode(ImageMode.RAW);
		this.imageReceiver = imageReceiver;
	}




	@Override
	public void stopRunning() {
		isRunning = false;
	}

	public void run() {
		int startAttempt = -1;
		while((! isRunning) && (startAttempt < 5)) {
			try {
				startAttempt++;
				frameGrabber.start();
				isRunning = true;
			} catch (Exception e2) {
				try {
					frameGrabber.restart();
					isRunning = true;
				} catch (Exception e3) {
					System.out.println("FlyCamera start attempt " +  startAttempt + " failed\n");
					e2.printStackTrace();
					e3.printStackTrace();

				}
			}

			while(isRunning){			
				try {
					imageReceiver.addImage(frameGrabber.grab());
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			try {
				frameGrabber.stop();
				frameGrabber.release();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}
