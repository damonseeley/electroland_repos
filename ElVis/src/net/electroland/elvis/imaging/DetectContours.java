package net.electroland.elvis.imaging;


import com.googlecode.javacpp.Loader;
import static com.googlecode.javacpp.Loader.*;

import static com.googlecode.javacv.cpp.opencv_core.cvCreateMemStorage;
import static com.googlecode.javacv.cpp.opencv_core.cvZero;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_RETR_EXTERNAL;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvFindContours;
import static com.googlecode.javacv.cpp.opencv_core.cvScalarAll;
import static com.googlecode.javacv.cpp.opencv_core.cvPoint;
import static com.googlecode.javacv.cpp.opencv_core.cvRect;
import static com.googlecode.javacv.cpp.opencv_core.cvDrawContours;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvBoundingRect;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvMoments;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvGetSpatialMoment;
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_core.cvRectangle;
import static com.googlecode.javacv.cpp.opencv_core.cvCircle;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvContour;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_imgproc.CvMoments;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.CvSeqBlock;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

import net.electroland.elvis.blobtracking.Blob;
import java.util.Vector;;

public class DetectContours {
	Vector<Blob> detectedBlobs; 

	float minBlobSize = 50;
	float maxBlobSize = 100;

	CvMemStorage  mem = null;
	CvSeq contours = null;
	int contourCnt;
	CvMoments moments = null;

	public DetectContours() {
		detectedBlobs = new Vector<Blob>();
		moments = new CvMoments();
		mem = CvMemStorage.create();// cvCreateMemStorage(0);//
		contours = new CvSeq();		
	}
	public void setMinBlobsize(float f) {
		minBlobSize = f;
	}
	public void setMaxBlobsize(float f) {
		minBlobSize = f;
	}

	public float getMinBlobSize() { return minBlobSize;}
	public float getMaxBlobSize() { return maxBlobSize;}
	//	public 	void detectContours(IplImage  src, IplImage dst) {

	public void detectBlobs() {
		if(contours != null) {
			Vector<Blob> newBlobs = new Vector<Blob>(detectedBlobs.size() + 10); // always be ready for a few more blobs than last time
			//TODO:  right now allocating new vector to avoid threading problems with tracker.  should probably reuse for speed (but need to be careful...)


			// not sure why I had to add the ! contour.isNull() -- didn't need it before
			for (CvSeq contour = contours; (contour != null) && (! contour.isNull()); contour = contour.h_next()) {	
				cvMoments(contour, moments, 0);
				double area = moments.m00();
				if((area > minBlobSize) && (area < maxBlobSize)) {
					double areaInv = 1.0/area;
					float x = (float) (cvGetSpatialMoment(moments, 1, 0) *areaInv);
					float y = (float) (cvGetSpatialMoment(moments, 0, 1) *areaInv);	    	
					CvRect boundbox = cvBoundingRect(contour, 0);
					newBlobs.add(new Blob(x,y,  boundbox.x(), boundbox.x()+boundbox.width(), boundbox.y(), boundbox.y()+boundbox.width(), (float)area));	            
				}
			}
			detectedBlobs = newBlobs;

		}
	} 

	public void drawBlobs(IplImage dst) {
		for(Blob b : detectedBlobs) {
			cvRectangle( dst , 
					cvPoint( b.minX, b.minY ), 
					cvPoint( b.maxX, b.maxY ), 
					cvScalar( 255,255, 255, 0 ), 1, 8, 0 );
			cvCircle(dst, cvPoint((int)b.centerX, (int)b.centerY), 2, cvScalar( 255,255, 255, 0 ), 1, 8, 0 );
		}


	}

	public 	void detectContours(IplImage src, IplImage dst) {
		cvCopy(src, dst);	 	
		cvFindContours(dst, mem, contours, 	Loader.sizeof(CvContour.class) , CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);		
	}

}
