package net.electroland.elvis.imaging;

import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.PlanarImage;
import javax.media.jai.RenderedOp;


public class BackgroundImage {

	int initialFrameSkip = 0; // some cameras have a lot of noise at startup (or fade in light iSight)  don't want to use bad background to start

	double[] adaptation = { .01 };
	double[] memory = { 1.0 - adaptation[0] };

	ParameterBlock pbAdapt = new ParameterBlock();
	ParameterBlock pbMemory = new ParameterBlock();
	ParameterBlock pbAdd = new ParameterBlock();


	BufferedImage background= null; 

	AffineTransform ident = new AffineTransform();

	public BackgroundImage() {		
	}

	public BackgroundImage(double adaptation, int frameSkip) {
		this.adaptation[0] = adaptation;
		this.memory[0] = 1.0-adaptation;
		initialFrameSkip = frameSkip;

		pbAdapt.add(this.adaptation);
		pbMemory.add(memory);
	}

	public void reset(int frameSkip) {
		initialFrameSkip = frameSkip;
		background = null;
	}
	/**
	 * 
	 * @param bi - should be assumes BufferedImage is of type TYPE_USHORT_GRAY
	 * @return
	 */public BufferedImage update(RenderedImage bi) {	
		 if(initialFrameSkip-- > 0) return null;
		 if(background == null)  {
			 background = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_USHORT_GRAY);
			 background.setData(bi.getData());
			 return background;
		 } else { 
			 if(adaptation[0] == 0) 	return background; // don't bother processing just use static background


			 pbAdapt.setSource(bi, 0);
			 PlanarImage newUpdate  = JAI.create("multiplyConst", pbAdapt);

			 pbMemory.setSource(background,0);
			 RenderedOp newBackground  = JAI.create("multiplyConst", pbMemory);


			 pbAdd.setSource(newUpdate,0);
			 pbAdd.setSource(newBackground,1);




			 RenderedOp newBG =JAI.create("add", pbAdd);

			 background.setData(newBG.getData());	

			 //	newBG.dispose();
			 //	newUpdate.dispose();
			 //	newBackground.dispose();

		 }
		 return background;
	 }

	 public double getAdaptation() {
		 return adaptation[0];
	 }
	 
	 public void setAdaptation(double d) {
		 adaptation[0] = d;
		 memory[0] = 1.0-d;
	 }
}
