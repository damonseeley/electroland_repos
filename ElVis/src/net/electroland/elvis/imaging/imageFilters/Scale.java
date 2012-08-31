package net.electroland.elvis.imaging.imageFilters;
//CV_INTER_CUBIC 
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_AREA;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;

import java.nio.IntBuffer;
import java.util.StringTokenizer;

import net.electroland.elvis.net.StringAppender;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.BoolParameter;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Scale extends Filter {
	
	boolean needsUpdate = true;
	IntParameter scaleXParam;
	IntParameter scaleYParam;
	BoolParameter isOn;
	IplImage smallImg;

	public Scale(int srcWidth, int srcHeight, String propPrefix, ElProps props) {
		super();
		isOn = new BoolParameter(propPrefix +"IsOn", false, props);
		parameters.add(isOn);

		scaleXParam = new IntParameter(propPrefix+"ScaleX", 1, 10, props);
		scaleXParam.setMinValue(0);
		parameters.add(scaleXParam);

		scaleYParam = new IntParameter(propPrefix+"ScaleY", 1, 10, props);
		scaleYParam.setMinValue(0);
		parameters.add(scaleYParam);


	}
	public IplImage getSmallImg() {
		return smallImg;
	}

	public void incParameter(int p) {
		super.incParameter(p);
		needsUpdate = true;
	}
	public void decParameter(int p) {
		super.decParameter(p);
		needsUpdate = true;
	}
	@Override // scale allocates dst itself
	public IplImage apply(IplImage src) {
		return process(src);
	}

	public IplImage process(IplImage src) {
		if(	needsUpdate) {
			if(isOn.getBoolValue()) {
				smallImg = IplImage.create(scaleXParam.getIntValue(),scaleYParam.getIntValue(), src.depth(),1);
			} 
			dst = IplImage.create(src.width(), src.height() , src.depth(),1);

			needsUpdate = false;
		}
		if(isOn.getBoolValue()) {
			cvResize(src, smallImg, CV_INTER_AREA);
			cvResize(smallImg, dst, CV_INTER_AREA );			
		} else {
			cvCopy(src, dst);
		}


		/*
		if(isOn.getBoolValue()) {
			ByteBuffer bb = dst.getByteBuffer();
			int cnt = 0;
			System.out.println("==");
			while(bb.hasRemaining()) {
				Short val =(short)(  (short) bb.get() & 0xff);
				System.out.println(cnt++ + " " +val);
			}
		}
		 */
		return dst;


	}
	

}
