package net.electroland.elvis.imaging;

//static imports
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_GAUSSIAN;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvSmooth;

import com.googlecode.javacv.cpp.opencv_core.CvArr;

public class Blur {
	int radius = 3;

	public Blur() {
	}
	public Blur(int radius) {
		this.radius = radius;	
	}

	public  void apply(CvArr  src, CvArr  dst) {
        cvSmooth(src, dst, CV_GAUSSIAN,radius);
		
	}

}
