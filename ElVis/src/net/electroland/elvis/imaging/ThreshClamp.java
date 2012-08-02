package net.electroland.elvis.imaging;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvThreshold;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_THRESH_TRUNC;
public class ThreshClamp {
	
	public static IplImage workingCopy =  null;

	
	public static final double WHITE = 65535;

	double  low = 2000;
	double  high = WHITE;
	double  map = WHITE;
	
	public ThreshClamp(double thresh) {
		setThreshold(thresh);
	}

	public void setHigh(double v) {
		high = v;
	}
	
	//TODO: not sure what the "map" is.  Must checks
	public void setVal(double v) {
		map = v;
	}

	public void setLow(double v) {
		low = v;
	}
	
	public void setThreshold(double v) {
		setLow(v);
	}

	public double getLow() {
		return low;
	}
	
	public void apply(IplImage src, IplImage dst) {
		if(workingCopy == null) workingCopy = dst.clone();
		cvThreshold(src, dst, low, high, CV_THRESH_TRUNC);	
	}
	
	
}
