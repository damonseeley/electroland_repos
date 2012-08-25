package net.electroland.elvis.imaging.imageFilters;

import java.util.Vector;


import net.electroland.elvis.util.parameters.Parameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;

public abstract class Filter {
	IplImage dst;

	public Vector<Parameter> parameters = new Vector<Parameter>();


	public abstract IplImage process(IplImage src);
	
	public IplImage apply(IplImage src) {
		
		dst = (dst == null) ? cvCreateImage(cvGetSize(src), src.depth(), src.nChannels()): dst;
		return process(src);
	}

	public Parameter getParameter(int p) {
		try {
			return parameters.get(p);
		} catch (Exception e) {
			return null;
		}
	}

	public void incParameter(int p) {
		try {
			parameters.get(p).inc();
		} catch (Exception e) {
		}
	}
	public void decParameter(int p) {
		try {
			parameters.get(p).dec();
		} catch (Exception e) {
		}
	}

	public IplImage getImage() {
		return dst;
	}



}
