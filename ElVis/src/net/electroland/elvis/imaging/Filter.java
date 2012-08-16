package net.electroland.elvis.imaging;

import java.util.Vector;

import net.electroland.elvis.util.parameters.Parameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public abstract class Filter {
	IplImage dst;

	Vector<Parameter> parameters = new Vector<Parameter>();


	public abstract IplImage apply(IplImage src);

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
