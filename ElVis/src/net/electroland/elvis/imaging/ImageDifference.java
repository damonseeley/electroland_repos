package net.electroland.elvis.imaging;
import static com.googlecode.javacv.cpp.opencv_core.cvAbsDiff;

import com.googlecode.javacv.cpp.opencv_core.CvArr;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class ImageDifference extends Filter{


	public IplImage apply(IplImage a, IplImage b) {
		dst = (dst == null) ? a.clone() : dst;
		cvAbsDiff(a,b, dst);	
		return dst;

	}

	public IplImage apply(IplImage src) {
		System.out.println("ImageDifference needs two sources");
		return null;
	}





}

