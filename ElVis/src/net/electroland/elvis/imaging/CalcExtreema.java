package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.cvMinMaxLoc;

import com.googlecode.javacv.cpp.opencv_core.CvArr;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;


public class CalcExtreema {

	public int sampleSize = 2;
	double[] minAr = new double[1];
	double[] maxAr = new double[1];
	CvPoint minLoc = new CvPoint();
	CvPoint maxLoc = new CvPoint();

	double maxVal = -1;
	double minVal = -1;

	public CalcExtreema() {
	}

	public void calc(CvArr img, CvArr roi) {
		cvMinMaxLoc(img, minAr, maxAr, minLoc, maxLoc, roi);
		minVal = minAr[0];
		maxVal = maxAr[0];
	}

	public double getMin() { return minVal; }
	public double getMax() { return maxVal; }
	public CvPoint getMinLoc() { return minLoc; }
	public CvPoint getMaxLoc() { return maxLoc; }
}
