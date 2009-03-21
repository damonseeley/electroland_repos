package net.electroland.elvis.imaging.acquisition.axisCamera;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

public class NoHoSouthCam extends AxisCamera {
	public static String url ="http://elnoho.dyndns.org:60/";
	
	public static String username = "noho";
	public static String password = "n";
	
	public NoHoSouthCam(int w, int h, ImageReceiver imageReceiver) {
		this(w,h,imageReceiver, false);
	}	

	public NoHoSouthCam(int w, int h, ImageReceiver imageReceiver, boolean color) {
		super(url, w, h, 0, color?1:0 , username, password, imageReceiver);
	}

}
