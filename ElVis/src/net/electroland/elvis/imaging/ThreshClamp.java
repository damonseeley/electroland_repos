package net.electroland.elvis.imaging;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvThreshold;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_THRESH_BINARY;

public class ThreshClamp {
	
	public static IplImage workingCopy =  null;

	public static final double WHITE = 65535;

	public static double thresholdValue;
	public static double clampVal = WHITE;
	
		
	public ThreshClamp(double thresh) {
		setThreshold(thresh);
	}

	public void setThreshold(double v) {
		thresholdValue = v;
	}

	public double getThreshold() {
		return thresholdValue;
	}
	
	public double getClampValue() {
		return clampVal;
	}
	public void setClampValue(double v) {
		clampVal = v;
	}
	
	public void apply(IplImage src, IplImage dst) {
		if(workingCopy == null) workingCopy = dst.clone();
		cvThreshold(src, dst, thresholdValue, clampVal, CV_THRESH_BINARY);	
	}
	
	
}
