package net.electroland.elvis.imaging;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class NoOpFilter extends Filter {

	@Override
	public IplImage apply(IplImage src) {
		dst = src;
		return src;
	}

}
