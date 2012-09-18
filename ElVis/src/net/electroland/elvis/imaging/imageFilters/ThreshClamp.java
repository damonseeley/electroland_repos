package net.electroland.elvis.imaging.imageFilters;

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
		threshParam = new DoubleParameter(prefPrefix+"threshold", 1,defThresh, props);
		threshParam.setRange(0, 255);
		parameters.add(threshParam);
	}


	public IplImage process(IplImage src) {
		cvThreshold(src, dst, threshParam.getDoubleValue(), clampVal, CV_THRESH_BINARY);	
		return dst;
	}


}
