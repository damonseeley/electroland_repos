package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.CV_32F;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvRemap;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.BoolParameter;
import net.electroland.elvis.util.parameters.DoubleParameter;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Unwarp extends Filter {

	public static CvScalar BLACK;

	boolean mapNeedsUpdate = true;

	public static int K1 = 0;
	public static int K2 = 1;
	public static int P1 = 2;
	public static int P2 = 3;
	public static int CENTERX = 4;
	public static int CENTERY = 5;
	public static int IS_ON = 6;

	IplImage mapx= null;
	IplImage mapy;

	CvMat matx;
	CvMat maty; 

	int width;
	int height;




	public Unwarp(int width, int height, ElProps props) {
		super();
		if(BLACK == null) {
			BLACK =cvScalar(0,0,0,0);
		}
		this.width =width;
		this.height = height;
		parameters.add(new DoubleParameter("unwarpK1", .0000001, 0,props));
		parameters.add(new DoubleParameter("unwarpK2", .000000000001, 0,props));
		parameters.add(new DoubleParameter("unwarpP1", .000001, 0,props));
		parameters.add(new DoubleParameter("unwarpP2", .000001, 0,props));
		parameters.add(new IntParameter("unwarpCenterX", 1, width/2,props));
		parameters.add(new IntParameter("unwarpCenterY", 1, height/2,props));
		parameters.add(new BoolParameter("unwarpIsOn", false,props));

	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}
	public void createMap() {
		mapNeedsUpdate = false;
		if(parameters.get(IS_ON).getBoolValue()) {
			double k1 = parameters.get(K1).getDoubleValue();
			double k2 = parameters.get(K2).getDoubleValue();
			double p1 = parameters.get(P1).getDoubleValue();
			double p2 = parameters.get(P2).getDoubleValue();
			double centerX = parameters.get(CENTERX).getDoubleValue();
			double centerY = parameters.get(CENTERY).getDoubleValue();
			createMap(k1,k2,p1,p2,centerX,centerY);
		} else {
			mapx = null;
		}
	}

	public void createMap(double k1, double k2, double p1, double p2, double centerx, double centery) {
		matx = CvMat.create(height, width, CV_32F);
		maty = CvMat.create(height, width, CV_32F);

		//		m = cvCreateImage( cvSize( width, height ), IPL_DEPTH_32F, 1 );
		//		mapy = cvCreateImage( cvSize( width, height ), IPL_DEPTH_32F, 1 );

		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				double dx = x-centerx;
				double dy = y-centery;
				double rsqr = dx*dx+dy*dy;
				double newX = x + dx * (k1*rsqr + k2*rsqr*rsqr) + p1*(rsqr + 2*dx*dx);
				double newY = y + dy * (k1*rsqr + k2*rsqr*rsqr) + p2*(rsqr + 2*dy*dy);
				matx.put(y, x, newX);
				maty.put(y, x, newY);
			}
		}

		mapx =  new IplImage (matx); 
		mapy =  new IplImage (maty); 

	}


	public void incParameter(int p) {
		super.incParameter(p);
		mapNeedsUpdate = true;
	}
	public void decParameter(int p) {
		super.decParameter(p);
		mapNeedsUpdate = true;
	}

	public IplImage apply(IplImage src) {
		dst = (dst == null) ? src.clone() : dst;
		if(mapNeedsUpdate) {
			createMap();
		}
		if(mapx == null) {
			cvCopy(src,dst);
		} else {
			cvRemap(src, dst, mapx, mapy,0,BLACK);		
		}
		return dst;
	}




}
