package net.electroland.elvis.imaging;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.RenderedOp;

public class ImageDifference {
	
	public static ParameterBlock pb = new ParameterBlock();


	public static void apply(RenderedImage a, RenderedImage b, BufferedImage dest) {
	
		
		 dest.setData(apply(a,b).getData());
	}
	

		
	public  static RenderedOp apply(RenderedImage a, RenderedImage b) {
		
		pb.setSource(a, 0);
		pb.setSource(b, 1);
		RenderedOp sub1 = JAI.create("subtract", pb);

		
		pb.setSource(b, 0);
		pb.setSource(a, 1);
		RenderedOp sub2 = JAI.create("subtract", pb);


		pb.setSource(sub1, 0);
		pb.setSource(sub2, 1);
		RenderedOp sum= JAI.create("add", sub1, sub2);
		
		
		return sum;

		
	}
}
