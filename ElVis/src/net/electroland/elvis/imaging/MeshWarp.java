package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.CV_64F;
import static com.googlecode.javacv.cpp.opencv_core.cvMatMul;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.IplImage;


public class MeshWarp extends Filter {

	@Override
	public IplImage apply(IplImage src) {
		return null; //UNDON

	}

	CvMat transformPoint( CvPoint pointToTransform,  CvMat matrix) {
		CvMat originVector = CvMat.create(3, 1, CV_64F);
		originVector.put(0, pointToTransform.x());
		originVector.put(1, pointToTransform.y());
		originVector.put(2, 1);
		CvMat transformedVector = CvMat.create(3, 1, CV_64F);
		transformedVector.put(0, pointToTransform.x());
		transformedVector.put(1, pointToTransform.y());
		transformedVector.put(2, 1);

		cvMatMul(matrix, originVector, transformedVector);
		
		double scale = 1.0/transformedVector.get(2);
		CvMat newPoint = CvMat.create(2, 1, CV_64F);
		newPoint.put(0, transformedVector.get(0)*scale);
		newPoint.put(1, transformedVector.get(1)*scale);

		return newPoint;
		}
}
