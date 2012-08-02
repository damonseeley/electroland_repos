package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.cvAvg;
import static com.googlecode.javacv.cpp.opencv_core.cvReleaseImage;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.image.BufferedImage;

import com.googlecode.javacv.cpp.opencv_core.CvArr;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class RoiAve {
	Shape roiShape;
	IplImage roi = null;	

	/*
	 * not sure if this is used
	 public RoiAve(Shape shape) {
		 roiShape;
	 }s
	 */


	public RoiAve() {
	}
	public void setRoi(Shape shape) {
		this.roiShape = shape;
	}
	
	public void setRoi(IplImage roi) {
		if(roi != null) {
			cvReleaseImage(roi);
		}
		roiShape = null;
		this.roi = roi;
	}
	
	public CvScalar getAverage(IplImage im, IplImage roi) {
		if(roiShape != null) {
			//TODO: this should be moved out of get average, also need to confirm mask colors are ok
			BufferedImage bi = new BufferedImage(im.width(), im.height(), BufferedImage.TYPE_USHORT_GRAY);
			Graphics2D g = ((Graphics2D)bi.getGraphics());
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, bi.getWidth(), bi.getHeight());
			g.setColor(Color.WHITE);
			g.fill(roiShape);
			roiShape = null;
			roi = IplImage.createFrom(bi);
		}
		return cvAvg(im, roi);

	}


	public double getAverage(CvArr im) {
		return cvAvg(im, roi).getVal(0);

	}
}
