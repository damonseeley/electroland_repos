package net.electroland.elvis.imaging;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;


public class BlobWriter extends Filter {
	DetectContours detectContours;

	public BlobWriter(DetectContours dc) {
		detectContours = dc;
	}
	@Override
	public IplImage apply(IplImage src) {
		dst = (dst == null) ? src.clone() : dst;
		cvCopy(src, dst);
		detectContours.drawBlobs(dst);		
		return dst;

	}


}
