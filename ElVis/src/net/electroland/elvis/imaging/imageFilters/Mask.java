package net.electroland.elvis.imaging.imageFilters;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvConvertImage;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;


public class Mask extends Filter {
	IplImage mask;

	public Mask(String imageName) {
		this(cvLoadImage(imageName));
	}
	public Mask(IplImage maskOrig) {
		super();
		if(maskOrig != null) {
			mask = IplImage.create(cvGetSize(maskOrig), 8, 1);
			cvConvertImage(maskOrig, mask, 0);
		}
	}

	public IplImage process(IplImage src) {
		if(mask==null) {
			dst = src;
		} else {
			cvCopy(src, dst, mask);
		} 
		return dst;
	}

}
