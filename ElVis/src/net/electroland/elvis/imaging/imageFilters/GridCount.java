package net.electroland.elvis.imaging.imageFilters;

import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvCountNonZero;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSubRect;
import static com.googlecode.javacv.cpp.opencv_core.cvScalarAll;
import static com.googlecode.javacv.cpp.opencv_core.cvSet;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.BoolParameter;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class GridCount extends Filter {

	CvRect rects[][]; 
	int areas[][];
	int totals[][];
	boolean needsUpdate = false;
	boolean isOn = false;; // need param here
	BoolParameter isOnParam;
	BoolParameter shouldRenderParam;
	boolean isRendering;
	int imgWidth;
	int imgHeight;
	IntParameter countGridWidth;
	IntParameter countGridHeight;
	int cellWidth;
	int cellHeight;
	int cols;
	int rows;
	public GridCount(int srcWidth, int srcHeight, ElProps props) {
		imgWidth = srcWidth;
		imgHeight = srcHeight;
		// add ison param here

		isOnParam  = new BoolParameter("grdCountIsOn", true);
		shouldRenderParam  = new BoolParameter("grdCountIsRendering", true);
		countGridWidth = new IntParameter("countGridWidth", 1, srcWidth / 10);
		countGridHeight = new IntParameter("countGridHeight", 1, srcHeight / 10);
		this.parameters.add(isOnParam);
		this.parameters.add(countGridWidth);
		this.parameters.add(countGridHeight);
		update();



	}

	public void update() {
		needsUpdate = false;
		isOn = isOnParam.getBoolValue();
		isRendering = shouldRenderParam.getBoolValue();
		if(! isOn) return;

		 cols = countGridWidth.getIntValue();
		 rows = countGridHeight.getIntValue();

		cellWidth = imgWidth / cols;
		cellHeight = imgHeight/rows;

		rects = new CvRect[cols][rows]; 
		areas = new int[cols][rows];
		totals = new int[cols][rows];

		int area = cellWidth * cellHeight;


		for(int i = 0; i < cols-1; i++) {
			for(int j = 0; j < rows-1; j++) {
				rects[i][j] = new CvRect(i*cellWidth, j*cellHeight, cellWidth, cellHeight);
				System.out.println(rects[i][j]);
				areas[i][j] = area;
			}
		}
		int i = cellWidth -1;
		int lastColWidth = imgWidth - (i*cellWidth);
		area = lastColWidth * cellHeight;
		for(int j = 0; j < rows-1; j++) {
			rects[i][j] = new CvRect(i*cellWidth, j*cellHeight, lastColWidth, cellHeight);
			areas[i][j] = area;
		}

		int j = cellHeight -1;
		int lastRowHeight = imgHeight - (j * cellHeight);
		area = lastRowHeight - (j*cellHeight); 
		for(i = 0; i < cols-1; i++) {
			rects[i][j] = new CvRect(i*cellWidth, j*cellHeight, cellWidth, lastRowHeight);
			areas[i][j] = area;
		}
		i = cols -1;
		j = rows -1;
		rects[i][j] = new CvRect(i*cellWidth, j*cellHeight, lastColWidth, lastRowHeight);
		areas[i][j] = lastRowHeight * lastColWidth;
	}
	public void incParameter(int p) {
		super.incParameter(p);
		needsUpdate = true;
	}
	public void decParameter(int p) {
		super.decParameter(p);
		needsUpdate = true;
	}

	@Override
	public IplImage process(IplImage src) {
		if(src == null) return null;
		if(needsUpdate) update();
		if(dst==null) dst = src.clone();
		if(! isRendering) {
			cvCopy(dst, src);
		}
		
		if(isOn) {
			for(int i =0; i < rects.length; i++) {
				for(int j = 0; j< rects[0].length; j++) {
					CvMat subimg = new CvMat();
					cvGetSubRect(src, subimg, rects[i][j]);
					int total = cvCountNonZero(subimg);
					totals[i][j] = cvCountNonZero(subimg);
					if(isRendering) {
						CvScalar scalar = cvScalarAll((int) (255 * (float)total/(float)areas[i][j]));
						cvSet(subimg, scalar);
					}
				}
			}
		} 
		return dst;
	}

}
