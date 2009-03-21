package net.electroland.elvis.imaging.acquisition.axisCamera;

import net.electroland.elvis.imaging.acquisition.ImageReceiver;

public class NavyCam extends AxisCamera {
	public static String url ="http://navystreet.dyndns.org:70/";
	public static int w = 160;
	public static int h = 120;
	
	public static String username = "n";
	public static String password = "n";
	
	public NavyCam(int w, int h, ImageReceiver imageReceiver) {
		this(w,h,imageReceiver, false);
	}	

	public NavyCam(int w, int h, ImageReceiver imageReceiver, boolean color) {
		super(url, w, h, 0, color?1:0 , username, password, imageReceiver);
	}

}
