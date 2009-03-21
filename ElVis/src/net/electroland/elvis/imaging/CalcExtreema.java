package net.electroland.elvis.imaging;

import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.ROI;
import javax.media.jai.RenderedOp;

public class CalcExtreema {
	ParameterBlock pb = new ParameterBlock();
	
	public int sampleSize = 2;
	double min = -1;
	double max = -1;

	public CalcExtreema() {
		pb.add(null);
		pb.add(sampleSize);
		pb.add(sampleSize);
	}
	
	public void calc(RenderedImage img, ROI roi) {
		 pb.setSource(img,0);  
		 pb.set(roi, 0); // roi is null check whole image
	     RenderedOp op = JAI.create("extrema", pb);
	     double[][] extrema = (double[][]) op.getProperty("extrema");
	     min = extrema[0][0];
	     max = extrema[1][0];
	}
	
	public double getMin() { return min; }
	public double getMax() { return max; }
}
