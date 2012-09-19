package net.electroland.elvis.imaging.imageFilters;

import static com.googlecode.javacv.cpp.opencv_core.CV_32FC1;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateMat;
import static com.googlecode.javacv.cpp.opencv_core.cvScalarAll;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_LINEAR;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_WARP_FILL_OUTLIERS;
import static com.googlecode.javacv.cpp.opencv_imgproc.cv2DRotationMatrix;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvWarpAffine;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.DoubleParameter;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvPoint2D32f;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Rotate extends Filter {
	boolean needsUpdate = true;
	DoubleParameter angleParam;
	IntParameter centerX;
	IntParameter centerY;
	IplImage dest;
	CvScalar fillval;
	CvMat mapMatrix;
	int flags;

	public Rotate(ElProps props) {
		super();
		angleParam = new DoubleParameter("rotateAngle", .5, 0, props);
		angleParam.setRange(-180, 180);
		parameters.add(angleParam);

		centerX = new IntParameter("rotateCenterOffsetX", 1, 0, props);
		parameters.add(centerX);
		centerY = new IntParameter("rotateCenterOffsetY", 1, 0, props);
		parameters.add(centerY);
	}


	@Override
	public IplImage process(IplImage src) {
		if(needsUpdate) {
			if(angleParam.getDoubleValue() == 0) {
				mapMatrix = null;

			} else {
				int x = (src.width() / 2) + centerX.getIntValue();
				int y = (src.height() / 2) + centerY.getIntValue();
				CvPoint2D32f center = new CvPoint2D32f(x,y);
				mapMatrix = cvCreateMat( 2, 3, CV_32FC1 );
				fillval = cvScalarAll(0);
				cv2DRotationMatrix(center, angleParam.getDoubleValue(), 1.0, mapMatrix);
				flags = CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS;
				//			cv2DRotationMatrix(center, angle, 1.0, mapMatrix);


			}
			if(mapMatrix == null) {
				cvCopy(src, dst);
			} else {
				cvWarpAffine(src, dst, mapMatrix, flags, fillval);
			}
		}

		return null;
	}

	public void incParameter(int p) {
		super.incParameter(p);
		needsUpdate = true;
	}
	public void decParameter(int p) {
		super.decParameter(p);
		needsUpdate = true;
	}

}
