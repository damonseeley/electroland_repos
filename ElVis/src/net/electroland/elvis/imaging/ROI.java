package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.cvRect;
import static com.googlecode.javacv.cpp.opencv_core.cvSetImageROI;
import static com.googlecode.javacv.cpp.opencv_core.cvResetImageROI;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class ROI {
	CvRect rect;
	public ROI(ElProps props) {
		this(props, "");
	}

	public ROI(ElProps props, String prefix) {
		this(
				props.getProperty(prefix+"roiLeft", 0),
				props.getProperty(prefix+"roiRight", 0),
				props.getProperty(prefix+"roiWidth", props.getProperty("srcWidth", 640)),
				props.getProperty(prefix+"roiHeight", props.getProperty("srcHeight", 480))
				);
	}
	public ROI(int left, int top, int w, int h) {
		rect = cvRect(left, top, w, h);
	}

	public void setForImage(IplImage img) {
		cvSetImageROI(img, rect);
	}

	public void removeForImage(IplImage img) {
		cvResetImageROI(img);
	}

}
