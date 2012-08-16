package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_imgproc.CV_THRESH_BINARY;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvThreshold;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.DoubleParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class ThreshClamp extends Filter{


	public static final double WHITE = 255;

	//	public static double thresholdValue;
	public static double clampVal = WHITE;
	DoubleParameter threshParam;

	public ThreshClamp(int defThresh, String prefPrefix, ElProps props) {
		super();
		threshParam = new DoubleParameter(prefPrefix+"threshold", defThresh, 100, props);
		parameters.add(threshParam);
	}

	/*
	public void setThreshold(double v) {
		thresholdValue = v;
	}

	public double getThreshold() {
		return thresholdValue;
	}
	 */
	/*
	public double getClampValue() {
		return clampVal;
	}
	public void setClampValue(double v) {
		clampVal = v;
	}*/

	public IplImage apply(IplImage src) {
		dst = (dst == null) ? src.clone() : dst;
		cvThreshold(src, dst, threshParam.getDoubleValue(), clampVal, CV_THRESH_BINARY);	
		return dst;
	}


}
