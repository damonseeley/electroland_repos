package net.electroland.elvis.imaging.acquisition.axisCamera;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

public class LocalCam extends AxisCamera {
	public static String url ="http://10.0.1.90/";
	
	public static String username = "root";
	public static String password = "n0h0";
	
	public LocalCam(int w, int h, ImageReceiver imageReceiver) {
		this(w,h,imageReceiver, false);
	}	

	public LocalCam(int w, int h, ImageReceiver imageReceiver, boolean color) {
		super(url, w, h, 0, color?1:0 , username, password, imageReceiver);
	}

}
