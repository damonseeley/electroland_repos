package net.electroland.elvis.imaging.acquisition.FlyCapture;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.FlyCaptureFrameGrabber;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.FrameGrabber.ImageMode;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

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
//		frameGrabber.setImageWidth(width);
//		frameGrabber.setImageHeight(height);
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
			
			try {
				IplImage img = frameGrabber.grab();
				if((img.width() != width) || (img.height() != height)) {
					System.out.println("WARNING: FlyCapture settings do not match requested image size:");
					System.out.println("WARNING:         reqeusted image size: " + width + "x" + height);
					System.out.println("WARNING:          received image size: " + img.width() + "x" + img.height());
					System.out.println("WARNING: Change camera settins in FlyCap2 application to fix");
				}
			} catch (Exception e1) {
				e1.printStackTrace();
				
			}
			
			

			while(isRunning){			
					try {
						imageReceiver.addImage(frameGrabber.grab());
//						Thread.sleep(1000/60);
					} catch(Exception e) {
					/*	
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					*/
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
