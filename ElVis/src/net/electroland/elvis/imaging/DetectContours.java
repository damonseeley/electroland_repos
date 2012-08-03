package net.electroland.elvis.imaging;

import static com.googlecode.javacpp.Loader.sizeof;
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
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_core.cvRectangle;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvContour;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.CvSeqBlock;
import com.googlecode.javacv.cpp.opencv_core.IplImage;


public class DetectContours {
	CvMemStorage  mem = null;
	CvSeq contours = null;
	int contourCnt;
	
	public DetectContours() {
		  mem = cvCreateMemStorage(0);
		 contours = new CvSeq();		
	}
	public void drawContours(IplImage  src, IplImage dst) {
		detectContours(src);
		cvZero(dst);
	    CvRect boundbox;
	    CvSeq ptr = new CvSeq();

	    for (ptr = contours; ptr != null; ptr = ptr.h_next()) {
	        boundbox = cvBoundingRect(ptr, 0);

	            cvRectangle( dst , cvPoint( boundbox.x(), boundbox.y() ), 
	                cvPoint( boundbox.x() + boundbox.width(), boundbox.y() + boundbox.height()),
	                cvScalar( 0, 255, 0, 0 ), 1, 0, 0 );
	    }
	}

	    /*
		cvDrawContours(
				dst,
				contours,
				cvScalarAll(255),
				cvScalarAll(255),
				100,
				1,
				8,
				cvPoint(0,0));}
		/*
		CvSeqBlock contour = contours.first();
		   int i = 0;
		while(contour != null) {
			/*
			CvSeq poly = cvApproxPoly(contour, sizeof(CvContour.class), mem, CV_POLY_APPROX_DP , 3,1);	
		
		   
			 double area = cvContourArea( contour,CV_WHOLE_SEQ,0);
			 System.out.println(i +": " + "area"  + area);
			 contour  = contour.h_next(); 
			 i++;
			 
			contour = contours.h_next().first();
		}
		    
			 
			
		}
//				CvSeq poly = cvApproxPoly(contour, sizeof(CvContour.class), mem, CV_POLY_APPROX_DP , 5,0);
				
				
	//			 cvPolyLine(dst, poly, poly.total(),1, 1, 255, 1, 8, 0);
	//		}
			
//			approxPolyDP(const Mat& curve, vector<Point>& approxCurve, double epsilon, bool closed)
			/*
			cvDrawContours(
				dst,
				contours,
				cvScalarAll(255),
				cvScalarAll(255),
				100,
				1,
				8,
				cvPoint(0,0));
		}
		*/
	
	public 	CvSeq detectContours(IplImage  src) {
		  CvMemStorage mem = CvMemStorage.create();
		    CvSeq contours = new CvSeq();

		    cvFindContours(src, mem, contours, 	Loader.sizeof(CvContour.class) , CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
		    if(contours!=null) {
		    System.out.println("totals"  + contours.total());
		    }
		    this.contours = contours;
		    this.mem = mem;
		return contours;
	}
      	
}
