package net.electroland.elvis.imaging.imageFilters;
//CV_INTER_CUBIC 
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_INTER_CUBIC;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvResize;

import java.nio.ByteBuffer;

import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.BoolParameter;
import net.electroland.elvis.util.parameters.IntParameter;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Scale extends Filter {
	boolean needsUpdate = true;
	IntParameter scaleXParam;
	IntParameter scaleYParam;
	BoolParameter isOn;

	public Scale(int srcWidth, int srcHeight, String propPrefix, ElProps props) {
		super();
		scaleXParam = new IntParameter(propPrefix+"ScaleX", 1, 30, props);
		scaleXParam.setMinValue(0);
		parameters.add(scaleXParam);

		scaleYParam = new IntParameter(propPrefix+"ScaleY", 1, 30, props);
		scaleYParam.setMinValue(0);
		parameters.add(scaleYParam);

		isOn = new BoolParameter(propPrefix +"IsOn", false, props);
		parameters.add(isOn);

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
				dst = IplImage.create(scaleXParam.getIntValue(),scaleYParam.getIntValue(), src.depth(),1);
			} else {
				dst = IplImage.create(src.width(), src.height() , src.depth(),1);
			}
			needsUpdate = false;
		}
		cvResize(src, dst, CV_INTER_CUBIC);

		if(isOn.getBoolValue()) {
			ByteBuffer bb = dst.getByteBuffer();
			int cnt = 0;
			System.out.println("==");
			while(bb.hasRemaining()) {
				Short val =(short)(  (short) bb.get() & 0xff);
				System.out.println(cnt++ + " " +val);
			}
		}
		return dst;


	}

}
