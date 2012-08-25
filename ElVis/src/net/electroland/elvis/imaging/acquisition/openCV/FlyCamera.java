package net.electroland.elvis.imaging.acquisition.openCV;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;

import com.googlecode.javacv.FlyCaptureFrameGrabber;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.FrameGrabber.ImageMode;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class FlyCamera extends OpenCVCam  {
	int width;
	int height;

	FlyCaptureFrameGrabber frameGrabber;
	ImageReceiver imageReceiver;
	boolean isRunning;

	public FlyCamera (ImageReceiver imageReceiver, int w, int h, int dev) throws Exception {
		super(imageReceiver, w,h , new FlyCaptureFrameGrabber(dev));
		frameGrabber.setImageMode(ImageMode.RAW);
	}
}