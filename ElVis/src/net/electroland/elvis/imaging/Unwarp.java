package net.electroland.elvis.imaging;

import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.IOException;

import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

public class Unwarp {
	IplImage mapx;
	IplImage mapy;
	

	int width;
	int height;
	
	

	double centerX;
	double centerY;
	
	double k1,k2;
	double p1,p2;
	
	public Unwarp(int width, int height) {
		this.width =width;
		this.height = height;
		
	}
	
	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
	public void createMap() {
		createMap(k1,k2,p1,p2,centerX,centerY);
	}
	public void createMap(double k1, double k2, double p1, double p2, double centerx, double centery) {
		CvMat matx = CvMat.create(height, width, CV_32F);
		CvMat maty = CvMat.create(height, width, CV_32F);
		
//		m = cvCreateImage( cvSize( width, height ), IPL_DEPTH_32F, 1 );
//		mapy = cvCreateImage( cvSize( width, height ), IPL_DEPTH_32F, 1 );
		
		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				double dx = x-centerX;
				double dy = y-centerY;
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
	public double getCenterX() {
		return centerX;
	}

	public void setCenterX(double centerX) {
		this.centerX = centerX;
	}

	public double getCenterY() {
		return centerY;
	}

	public void setCenterY(double centerY) {
		this.centerY = centerY;
	}

	public double getK1() {
		return k1;
	}

	public void setK1(double k1) {
		this.k1 = k1;
	}

	public double getK2() {
		return k2;
	}

	public void setK2(double k2) {
		this.k2 = k2;
	}

	public double getP1() {
		return p1;
	}

	public void setP1(double p1) {
		this.p1 = p1;
	}

	public double getP2() {
		return p2;
	}

	public void setP2(double p2) {
		this.p2 = p2;
	}
	public void apply(IplImage src, IplImage dst) {
		if(mapx == null) dst = src.clone();
		cvRemap(src, dst, mapx, mapy,0,cvScalar(0,0,0,0));		
	}
	

	
	
}
