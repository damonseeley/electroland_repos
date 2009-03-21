package net.electroland.elvis.imaging;

import java.awt.Shape;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.ROI;
import javax.media.jai.ROIShape;

public class RoiAve {
	ROI roi;
	 ParameterBlock pb = new ParameterBlock();
	 
	 public RoiAve(Shape shape) {
		 this(new ROIShape(shape));
	 }

	 public RoiAve(ROI roi) {
	     pb.add(roi);       // null ROI means whole image
	     pb.add(1);          // check every pixel horizontally
	     pb.add(1);          // check every pixel vertically
	}
	
	public double getAverage(RenderedImage im) {
	     pb.setSource(im, 0);
	     RenderedImage meanImage = JAI.create("mean", pb, null);
	     double[] mean = (double[])meanImage.getProperty("mean");
	     return mean[0];
	}
}
