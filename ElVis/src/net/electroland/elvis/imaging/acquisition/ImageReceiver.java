package net.electroland.elvis.imaging.acquisition;

import java.awt.image.BufferedImage;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public interface ImageReceiver {
	public void addImage(IplImage i);
	public void addImage(BufferedImage i);
	public void receiveErrorMsg(Exception cameraException);
}
