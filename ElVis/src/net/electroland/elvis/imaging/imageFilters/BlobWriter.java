package net.electroland.elvis.imaging.imageFilters;


import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;


public class BlobWriter extends Filter {
	DetectContours detectContours;

	public BlobWriter(DetectContours dc) {
		detectContours = dc;
	}
	@Override
	public IplImage process(IplImage src) {
		cvCopy(src, dst);
		detectContours.drawBlobs(dst);		
		return dst;

	}


}
