package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvAvg;
import static com.googlecode.javacv.cpp.opencv_core.cvCountNonZero;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_core.cvReleaseImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvConvertImage;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.image.BufferedImage;

import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class RoiAve {

	//TODO: can be spead up with roi for ave
	Shape roiShape;
	IplImage mask = null;	
	IplImage scratch = null;	

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
		if(mask != null) {
			cvReleaseImage(mask);
			mask = null;
		}
	}

	public void setRoi(IplImage roi) {
		if(mask != null) {
			cvReleaseImage(mask);
		}
		roiShape = null;
		this.mask = roi;
	}

	public double getAverage(IplImage im, IplImage curMask) {
		if((curMask == null) && (roiShape != null)) {
			//TODO: this should be moved out of get average, also need to confirm mask colors are ok
			BufferedImage bi = new BufferedImage(im.width(), im.height(), BufferedImage.TYPE_USHORT_GRAY);
			Graphics2D g = ((Graphics2D)bi.getGraphics());
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, bi.getWidth(), bi.getHeight());
			g.setColor(Color.WHITE);
			g.fill(roiShape);
			roiShape = null;
			if(mask != null) {
				cvReleaseImage(mask);
			}
			IplImage maskOrig = IplImage.createFrom(bi);
			mask = IplImage.create(cvGetSize(maskOrig), 8, 1);
			cvConvertImage(maskOrig, mask, 0);


		}

		return cvAvg(im, mask).getVal(0);
	}


	public double getAverage(IplImage im) {
		return getAverage(im, mask);
	}
}
