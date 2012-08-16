package net.electroland.elvis.imaging;

//static imports
import static com.googlecode.javacv.cpp.opencv_imgproc.cvErode;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Erode  extends Filter {
	IntParameter iterationsParam;

	public Erode(int defIterations, ElProps props) {
		super();
		iterationsParam = new IntParameter("erodeIterations", 1, defIterations, props);
		iterationsParam.setMinValue(0);
		parameters.add(iterationsParam);

	}



	@Override
	public IplImage apply(IplImage src) {
		dst = (dst == null) ? src.clone() : dst;
		cvErode(src, dst,null,iterationsParam.getIntValue());
		return dst;
	}
}
