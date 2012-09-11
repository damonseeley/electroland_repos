package net.electroland.elvis.util;

import java.io.IOException;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;
import net.electroland.elvis.imaging.acquisition.axisCamera.FlowerCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.LocalCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NavyCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoSouthCam;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvis.imaging.acquisition.openCV.FlyCamera;
import net.electroland.elvis.imaging.acquisition.openCV.OpenCVCam;

import com.googlecode.javacv.FrameGrabber.Exception;

public class CameraFactory {
	public static final String NAVY_SRC = "Navy St.";
	public static final String FLOWER_SRC = "Flower St.";
	public static final String NOHOSOUTH_SRC = "NoHo South";
	public static final String NOHONORTH_SRC = "NoHo North";
	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="Local Axis";
	public static final String FLY_SRC = "FlyCam";
	public static final String OPENCV_SRC = "OpenCV";
	
	public static ImageAcquirer camera(String s, int w, int h, ImageReceiver ir) throws IOException, Exception {
		if(s.equals(NAVY_SRC)) {
			return new NavyCam(w,h,ir, false);
		} else if(s.equals(FLOWER_SRC)) {
			return new FlowerCam(w,h,ir, false);
		} else if(s.equals(NOHOSOUTH_SRC)) {
			return new NoHoSouthCam(w,h,ir, false);
		} else if(s.equals(NOHONORTH_SRC)) {
			return new NoHoNorthCam(w,h,ir, false);
		} else if(s.equals(JMYRON_SRC)) {
			return new WebCam(w, h, 12, ir, false);
		} else if(s.equals(LOCALAXIS_SRC)) {
			return new LocalCam(w,h,ir, false);
		} else if(s.equals(FLY_SRC)) {
				return new FlyCamera(ir, w, h, 0);
		} else if(s.equals(OPENCV_SRC)){
			return new OpenCVCam(ir, w, h, 0);
		}else {
			System.out.println("Failed to resolve source: " + s.toString());
			throw new IOException("Unknown source");
			
		}
	}
}
