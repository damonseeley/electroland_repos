package net.electroland.elvis.imaging.imageFilters;

//static imports
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_GAUSSIAN;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvSmooth;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.OddParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Copy extends Filter {
	public Copy() {
		super();

	}


	@Override
	public IplImage process(IplImage src) {
		cvCopy(src,dst);
		return dst;
	}

}
