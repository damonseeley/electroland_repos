package net.electroland.elvis.imaging;

import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.PlanarImage;
import javax.swing.JFrame;

import com.sun.media.jai.widget.DisplayJAI;

public class ElvisImage {
	PlanarImage bi;


	public ElvisImage()  {
	}

	
	
	/**
	 * subtracts bi from this 
	 * @param bi
	 */
	
	public static PlanarImage ColorToGray(PlanarImage i) {
	   // RenderingHints rh =
        //    new RenderingHints(JAI.KEY_REPLACE_INDEX_COLOR_MODEL, i);
		double[][] matrix = { {0.114D, 0.587D, 0.299D, 0.0D} };

		if (i.getSampleModel().getNumBands() == 1) {
			System.out.println("already one band");
			return i;
		} else if (i.getSampleModel().getNumBands() != 3) {
			throw new IllegalArgumentException("Image # bands <> 3");
		}

		ParameterBlock pb = new ParameterBlock();
		pb.addSource(i);
		pb.add(matrix);
		return ( (PlanarImage) JAI.create("bandcombine", pb, null));
	} 
	
	public void subtract(PlanarImage otherImage) {
//		Graphics2D g = bi.createGraphics();
//		g.setComposite(comp)
	}

	public static void main(String args[]) {

		PlanarImage bgImage = ColorToGray(JAI.create("fileload", "images/tmpImg.gif"));
		PlanarImage image = ColorToGray(JAI.create("fileload", "images/tmpImg2.gif"));
		
		long t = System.currentTimeMillis() + 1000;
	     PlanarImage add = null;
	     int i = 0;
		while(t > System.currentTimeMillis()) {
			i++;
	     PlanarImage sub1 = (PlanarImage)JAI.create("subtract", image, bgImage);
	     PlanarImage sub2 = (PlanarImage)JAI.create("subtract", bgImage, image);
	  
	      add = (PlanarImage)JAI.create("add", sub1, sub2);
		  
		}
		  System.out.println( i + "frames per sec");
	     JFrame frame = new JFrame("foo");
	     
	     DisplayJAI display = new DisplayJAI(add);
	     frame.add(display);
	     frame.setVisible(true);
		
		
		
		
		
	}
}
