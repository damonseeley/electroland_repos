package net.electroland.elvis.imaging;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.RenderedOp;

public class ThreshClamp {
	public static final double WHITE = 65535;
	ParameterBlock pb = new ParameterBlock();

	double [] low = {2000};
	double [] high = {WHITE};
	double [] map = {WHITE};
	
	public ThreshClamp(double thresh) {
		low[0] = thresh;
		pb.add(low);
		pb.add(high);
		pb.add(map);
	}

	public void setHigh(double v) {
		high[0] = v;
	}
	public void setVal(double v) {
		map[0] = v;
	}

	public void setLow(double v) {
		low[0] = v;
	}
	
	public void setThreshold(double v) {
		setLow(v);
	}

	public double getLow() {
		return low[0];
	}
	public RenderedOp apply(RenderedImage src) {
		pb.setSource(src, 0);
		return JAI.create("threshold", pb);
	}
	public void apply(RenderedImage src, BufferedImage dst) {
	
		dst.setData(apply(src).getData());
	}
}
