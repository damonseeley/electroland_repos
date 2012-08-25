package net.electroland.elvis.imaging.imageFilters;
import static com.googlecode.javacv.cpp.opencv_core.cvAbsDiff;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;

import com.googlecode.javacv.cpp.opencv_core.CvArr;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class ImageDifference extends Filter{


	public IplImage apply(IplImage a, IplImage b) {
		if(dst == null) {
			dst = cvCreateImage(a.cvSize(), a.depth(), a.nChannels());
		} else {
			cvAbsDiff(a,b, dst);				
		}
		return dst;

	}

	public IplImage process(IplImage src) {
		System.out.println("ImageDifference needs two sources");
		return null;
	}





}

