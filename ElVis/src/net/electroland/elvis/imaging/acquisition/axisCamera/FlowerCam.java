package net.electroland.elvis.imaging.acquisition.axisCamera;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

public class FlowerCam extends AxisCamera {
	public static String url ="http://11flower.dyndns.org/";
	
	public static String username = "root";
	public static String password = "11fl0wer";
	
	public FlowerCam(int w, int h, ImageReceiver imageReceiver) {
		this(w,h,imageReceiver, false);
	}	

	public FlowerCam(int w, int h, ImageReceiver imageReceiver, boolean color) {
		super(url, w, h, 0, color?1:0 , username, password, imageReceiver);
	}

}
