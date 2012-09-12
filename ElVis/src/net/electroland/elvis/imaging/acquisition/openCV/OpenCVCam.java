package net.electroland.elvis.imaging.acquisition.openCV;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.FrameGrabber;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.FrameGrabber.ImageMode;
import com.googlecode.javacv.OpenCVFrameGrabber;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class OpenCVCam extends Thread implements ImageAcquirer {
	int width;
	int height;

	long timeForNextGrab = -1;
	float maxFPS;
	long delay;

	FrameGrabber frameGrabber;
	ImageReceiver imageReceiver;
	boolean isRunning;


	public OpenCVCam (ImageReceiver imageReceiver, int w, int h, int dev) throws Exception {
		this(imageReceiver, w, h, new OpenCVFrameGrabber(dev));
		frameGrabber.setImageMode(ImageMode.GRAY);
		setFPS(15);// seems like thinks don't work well if you grab w/o limits
	}

	public void setFPS(float fps) {
		maxFPS= fps;
		delay = (long) (1000.0f / maxFPS);
	}

	public float getFPS() {
		return maxFPS;
	}

	public OpenCVCam (ImageReceiver imageReceiver, int w, int h, FrameGrabber grabber) throws Exception {
		this.width = w;
		this.height = h;
		frameGrabber = grabber;
		frameGrabber.setImageWidth(width);
		frameGrabber.setImageHeight(height);
		this.imageReceiver = imageReceiver;
	}




	@Override
	public void stopRunning() {
		isRunning = false;
		synchronized(this) {
			this.notify(); // stop waiting
		}
	}

	public void run() {
		int startAttempt = 0;
		while((! isRunning) && (startAttempt < 5)) {
			try {
				frameGrabber.start();
				isRunning = true;
			} catch (Exception e2) {
				try {
					startAttempt++;
					frameGrabber.restart();
					isRunning = true;
				} catch (Exception e3) {
					System.out.println("FrameGrabber start attempt " +  startAttempt + " failed\n");
					e2.printStackTrace();
					e3.printStackTrace();

				}
			}
		}

		if(isRunning) {
			try {
				IplImage img = frameGrabber.grab();
				if((img.width() != width) || (img.height() != height)) {
					System.out.println("WARNING: FrameGrabber settings do not match requested image size:");
					System.out.println("WARNING:         reqeusted image size: " + width + "x" + height);
					System.out.println("WARNING:          received image size: " + img.width() + "x" + img.height());
					System.out.println("WARNING: Change camera settins in FlyCap2 application if possible");
				}
			} catch (Exception e1) {
				e1.printStackTrace();

			}
		}

		//Seems to need a small delay before grabbing
		if(delay > 0) {
		synchronized(this) {
			try {
				this.wait(delay);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		}


		while(isRunning){			
			long curTime = System.currentTimeMillis();
			long diff = timeForNextGrab  - curTime;
			if(diff > 0) {
				try {
					synchronized(this) {
						this.wait(diff);
					}
				} catch (InterruptedException e1) {
					if(! isRunning) {
						break;
					}
				}
			}
			timeForNextGrab = curTime +  delay;

			try {
				imageReceiver.addImage(frameGrabber.grab());
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
