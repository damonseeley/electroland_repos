package net.electroland.elvis.imaging;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.RenderedOp;

public class ImageDifference {

	public static ParameterBlock pbSub1 = new ParameterBlock();
	public static ParameterBlock pbSub2 = new ParameterBlock();
	public static ParameterBlock pbSum = new ParameterBlock();

	public static RenderedOp sub1 =  null;
	public static RenderedOp sub2 =  null;
	public static RenderedOp sum =  null;



	public static void apply(RenderedImage a, RenderedImage b, BufferedImage dest) {


		dest.setData(apply(a,b).getData());
	}



	public  static RenderedOp apply(RenderedImage a, RenderedImage b) {
		if (sub1 == null) {
			pbSub1.addSource(a);
			pbSub1.addSource(b);
			pbSub2.addSource(b);
			pbSub2.addSource(a);

			sub1 = JAI.create("subtract", pbSub1);
			sub2 = JAI.create("subtract", pbSub2);
			sum = JAI.create("add", sub1, sub2);

			return sum;
		} else {
			sub1.setSource(a, 0);
			sub1.setSource(b, 1);
			sub2.setSource(b, 0);
			sub2.setSource(a, 1);
			sub1.getNewRendering();
			sub2.getNewRendering();
//			sum.getNewRendering();


			return sum;
		}

	}
}
