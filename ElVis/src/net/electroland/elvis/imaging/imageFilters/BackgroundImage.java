package net.electroland.elvis.imaging.imageFilters;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_32F;
import static com.googlecode.javacv.cpp.opencv_core.cvAddWeighted;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_highgui.cvConvertImage;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.DoubleParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;


public class BackgroundImage extends Filter {

	IplImage internalModel; 
	IplImage internalSrc; 
	
	int initialFrameSkip = 0; // some cameras have a lot of noise at startup (or fade in light iSight)  don't want to use bad background to start

	public DoubleParameter adaptionParameter;

	//	double adaptation =  .01 ;
	//	double memory ;







	public BackgroundImage(double defAdaptation, int frameSkip, ElProps props) {
		super();
		adaptionParameter = new DoubleParameter("backgroundImageAdaption", .001, defAdaptation, props);
		parameters.add(adaptionParameter);
		initialFrameSkip = frameSkip;

	}

	public void reset(int frameSkip) {
		initialFrameSkip = frameSkip;
		dst = null;
	}


	public int getRemainingFrameSkip() {
		return initialFrameSkip;
	}



	/**
	 * 
	 * @param bi - should be assumes BufferedImage is of type TYPE_USHORT_GRAY
	 * @return
	 */

	@Override
	public IplImage process(IplImage im) {
		return update(im);
	}
	
	public IplImage apply(IplImage src) { 
		// don't want super.apply creating new image if dst == null
		return update(src);
	}



	public IplImage update(IplImage bi) {	
		if(initialFrameSkip-- > 0) return null;
		if(dst == null)  {
			dst = cvCreateImage(cvGetSize(bi), bi.depth(), bi.nChannels());
			internalSrc = cvCreateImage(cvGetSize(bi), IPL_DEPTH_32F, bi.nChannels());
			internalModel = cvCreateImage(cvGetSize(bi), IPL_DEPTH_32F, bi.nChannels());
			cvConvertImage(bi, internalModel,0);
			cvCopy(bi, dst);
			return dst;
		} else { 
			double adaptation =adaptionParameter.getDoubleValue();
			if(adaptation == 0) 	return dst; // don't bother processing just use static background
			double memory = 1.0 - adaptation;
			cvConvertImage(bi, internalSrc,0);
			cvAddWeighted(internalModel, memory, internalSrc, adaptation, 0, internalModel);
			cvConvertImage(internalModel, dst,0);
			

		}
		return dst;
	}


}
