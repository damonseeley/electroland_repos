package net.electroland.elvis.imaging;


import static com.googlecode.javacv.cpp.opencv_core.cvCircle;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvPoint;
import static com.googlecode.javacv.cpp.opencv_core.cvRectangle;
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_RETR_EXTERNAL;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvBoundingRect;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvFindContours;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvGetSpatialMoment;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvMoments;

import java.util.Vector;

import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_core.CvContour;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_imgproc.CvMoments;

public class DetectContours extends Filter {
	Vector<Blob> detectedBlobs; 

	IntParameter minBlobSizeParam;
	IntParameter maxBlobSizeParam;

	//	float minBlobSize;
	//	float maxBlobSize;

	CvMemStorage  mem = null;
	CvSeq contours = null;
	int contourCnt;
	CvMoments moments = null;
	ElProps props;

	public DetectContours(ElProps props) {
		super();
		this.props  = props;
		minBlobSizeParam = new IntParameter("minBlobSize", 1, 50, props);
		this.parameters.add(minBlobSizeParam);

		maxBlobSizeParam = new IntParameter("maxBlobSize", 1, 100, props);
		this.parameters.add(maxBlobSizeParam);

		detectedBlobs = new Vector<Blob>();
		moments = new CvMoments();
		mem = CvMemStorage.create();// cvCreateMemStorage(0);//
		contours = new CvSeq();		
	}


	public double getAverageBlobSize() {
		double acc= 0;
		double cnt  =0;
		for(Blob b : detectedBlobs) {
			acc += b.getSize();
			cnt+=1.0;
		}
		return acc/cnt;
	}
	public void detectBlobs() {
		if(contours != null) {
			int minBlobSize = minBlobSizeParam.getIntValue();
			int maxBlobSize = maxBlobSizeParam.getIntValue();
			Vector<Blob> newBlobs = new Vector<Blob>(detectedBlobs.size() + 10); // always be ready for a few more blobs than last time
			//TODO:  right now allocating new vector to avoid threading problems with tracker.  should probably reuse for speed (but need to be careful...)


			// not sure why I had to add the ! contour.isNull() -- didn't need it before
			for (CvSeq contour = contours; (contour != null) && (! contour.isNull()); contour = contour.h_next()) {	

				cvMoments(contour, moments, 0);
				double area = moments.m00();
				//				double area = cvContourArea(contour, CV_WHOLE_SEQ);
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

	/*
	public 	void detectContours(IplImage src, IplImage dst) {
		cvCopy(src, dst);	 	
		cvFindContours(dst, mem, contours, 	Loader.sizeof(CvContour.class) , CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);		
	}
	 */
	@Override
	public IplImage apply(IplImage src) {
		if(dst == null) dst = src.clone();
		cvCopy(src, dst);	 	
		cvFindContours(dst, mem, contours, 	Loader.sizeof(CvContour.class) , CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);		
		return dst;

	}

}
