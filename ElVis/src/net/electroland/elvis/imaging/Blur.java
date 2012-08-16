package net.electroland.elvis.imaging;

//static imports
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_GAUSSIAN;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvSmooth;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.OddParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Blur extends Filter {
	OddParameter radiusParam; 
	public Blur(int defRadius, ElProps props) {
		super();
		radiusParam = new OddParameter("blurRadius", 2, defRadius, props);
		radiusParam.setMinValue(0);
		parameters.add(radiusParam);

	}



	@Override
	public IplImage apply(IplImage src) {
		dst = (dst == null) ? src.clone() : dst;

		int radius = radiusParam.getIntValue();
		if(radius <= 0)  {
			cvCopy(src,dst);
		} else {
			cvSmooth(src, dst, CV_GAUSSIAN, radius);		
		}
		return dst;
	}

}
