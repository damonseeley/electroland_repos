package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.cvAddWeighted;

import com.googlecode.javacv.cpp.opencv_core.IplImage;



public class BackgroundImage {

	int initialFrameSkip = 0; // some cameras have a lot of noise at startup (or fade in light iSight)  don't want to use bad background to start


	double adaptation =  .01 ;
	double memory =  1.0 - adaptation;



	IplImage background= null; 


	public BackgroundImage() {		
	}
	

	public BackgroundImage(double adaptation, int frameSkip) {
		this.adaptation = adaptation;
		this.memory= 1.0-adaptation;
		initialFrameSkip = frameSkip;

	}

	public void reset(int frameSkip) {
		initialFrameSkip = frameSkip;
		background = null;
	}
	
	
	public int getRemainingFrameSkip() {
		return initialFrameSkip;
	}

	
	
	/**
	 * 
	 * @param bi - should be assumes BufferedImage is of type TYPE_USHORT_GRAY
	 * @return
	 */
	public IplImage update(IplImage bi) {	
		 if(initialFrameSkip-- > 0) return null;
		 if(background == null)  {
			 background = bi.clone();	
			 return background;
		 } else { 
			 if(adaptation == 0) 	return background; // don't bother processing just use static background
			 cvAddWeighted(background, memory, bi, adaptation, 0, background);
			 /*
			 cvScall
			 
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
*/
		 }
		 return background;
	 }

	 public double getAdaptation() {
		 return adaptation;
	 }
	 
	 public void setAdaptation(double d) {
		 adaptation = d;
		 memory = 1.0-d;
	 }
}
