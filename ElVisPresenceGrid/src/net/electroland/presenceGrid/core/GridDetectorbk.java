package net.electroland.presenceGrid.core;

import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.RenderedOp;

import net.electroland.elvis.imaging.BackgroundImage;
import net.electroland.elvis.imaging.ImageConversion;
import net.electroland.elvis.imaging.ImageDifference;
import net.electroland.elvis.imaging.ImageProcessor;
import net.electroland.elvis.imaging.ThreshClamp;

public class GridDetectorbk extends ImageProcessor {

	public boolean convertFromColor = false;

	public static enum ImgReturnType { RAW, GRAY, BGRND, DIFF, THRESH, MIPMAP };

	ImgReturnType imgReturnType = ImgReturnType.RAW;
	int mipmapLevel = 0;

	boolean rescaleMinMap = true;


	RenderedOp cropOp;
	RenderedOp scaleOp;

	BufferedImage grayImage;
	BackgroundImage background;
	BufferedImage diffImage;
	BufferedImage threshImage;
	BufferedImage threshImage2;
//	BufferedImage mipmapLevelImage;
//	ImageMIPMap mipmap;

	ThreshClamp thresh = new ThreshClamp(2000);
	ThreshClamp thresh2 = new ThreshClamp(2000);

	ImageConversion imageConversion = new ImageConversion();
	
//	AffineTransform downScaleTransform;
//	AffineTransform upscaleDisplayTransform; // just used if rendering to screen

	
	int inCropWidth;
	int inCropHeight;

	// start out with fullsize image
	// convert to grey
	// crop and scale to 
	// do image analysis
	//

	// added by DS 2012 for comaptibility with later ElVis build
	public void receiveErrorMsg(Exception e) {
		System.out.println(e);
	}

	
	public GridDetectorbk(int srcWidth, int srcHeight, int xOffset, int yOffset, int inCropWidth, int inCropHeight, int outputWidth, int outputHeight) {
		super(outputWidth, outputHeight);
		
		this.inCropWidth = inCropWidth;
		this.inCropHeight = inCropHeight;
		
		grayImage = new BufferedImage(inCropWidth,inCropHeight,BufferedImage.TYPE_USHORT_GRAY);
		background = new BackgroundImage(.001, 60);
		diffImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		threshImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		threshImage2 = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
	//	mipmapLevelImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);

//		downScaleTransform= AffineTransform.getScaleInstance(.5, .5);
		
		ParameterBlock cropPB = new ParameterBlock();
		cropPB.addSource(grayImage);
		cropPB.add((float)xOffset);
		cropPB.add((float)yOffset);
		cropPB.add((float)inCropWidth);
		cropPB.add((float)inCropHeight);
		cropOp = JAI.create("crop",cropPB);
		
		ParameterBlock scalePB = new ParameterBlock();
		float  xScale = (float) outputWidth / (float) inCropWidth;
		float  yScale = (float) outputHeight / (float) inCropHeight;
		scalePB.add(xScale);
		scalePB.add((float) outputHeight / (float) inCropHeight);
		scalePB.add(-xScale * (float) xOffset); // translate cropped section to 0,0
		scalePB.add(-yScale * (float) yOffset); 
		scalePB.addSource(cropOp);
		scaleOp = JAI.create("scale",scalePB);
		
		
	}



	public void setThresh(double d) {
		thresh.setLow(d);
		thresh.setHigh(65535);
		thresh.setVal(65535);
		thresh2.setLow(0);
		thresh2.setHigh(d);
		thresh2.setVal(0);
	}

	public double getThresh() {
		return thresh.getLow();
	}

	public void resetBackground(int frameSkip) {
		background.reset(frameSkip);
		resetFPSCalc();
	}

	public void setImageReturn(ImgReturnType ret) {
		imgReturnType = ret;
	}

	public void nextImageReturnType() {
		int nextOrd = imgReturnType.ordinal() + 1;
		if(nextOrd >= ImgReturnType.values().length) {
			nextOrd = 0;
		}
		imgReturnType =  ImgReturnType.values()[nextOrd];
		System.out.println("Displaying " + imgReturnType);
	}

	public void prevImageReturnType() {
		int prevOrd = imgReturnType.ordinal() - 1;
		if(prevOrd < 0) {
			prevOrd = ImgReturnType.values().length-1;
		}
		imgReturnType =  ImgReturnType.values()[prevOrd];
		System.out.println("Displaying " + imgReturnType);
	}
	
	public void incMipMapLevel() {
		mipmapLevel++;
		rescaleMinMap = true;
		System.out.println("minmap level" + mipmapLevel);
	}
	
	public void decMipMapLevel() {
		mipmapLevel--;
		mipmapLevel = (mipmapLevel < 0) ? 0 : mipmapLevel;
		rescaleMinMap = true;
		System.out.println("minmap level" + mipmapLevel);		
	}

	@Override
	public BufferedImage process(BufferedImage img) {
		if(convertFromColor) {
			imageConversion.convertFromRGB(img, grayImage);			
		} else {
			imageConversion.convertFromGray(img, grayImage);
		}

		BufferedImage bkImage = background.update(grayImage);
		
		if(bkImage == null) return null;

		ImageDifference.apply(bkImage, grayImage, diffImage);





		thresh.apply(diffImage, threshImage);
		thresh2.apply(threshImage, threshImage2);

	//	mipmap = getMipMap(threshImage2);
		
		BufferedImage returnImage = null;

		
		switch(imgReturnType) {
		case RAW:
			returnImage = img;
			break;
		case GRAY: 
			returnImage = grayImage;
			break;
		case BGRND: 
			returnImage = bkImage;
			break;
		case DIFF: 
			returnImage = diffImage;
			break; 
		case THRESH: 
			returnImage = threshImage2;
			break;
		case MIPMAP:
			
			cropOp.setSource(img, 0);
			returnImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
			returnImage.createGraphics().drawRenderedImage(scaleOp, new AffineTransform());
//			System.out.println(scaleOp.getWidth());
			
//			  returnImage = mipmapLevelImage;
			/*
			  ParameterBlock pb = new ParameterBlock();
	          pb.addSource(img);                   // The source image
	          pb.add(.5f);                        // The xScale
	          pb.add(.5f);                        // The yScale
	          pb.add(0.0F);                       // The x translation
	          pb.add(0.0F);                       // The y translation
	          pb.add(Interpolation.getInstance( Interpolation.INTERP_BILINEAR)); // The interpolation
	          
	          
	          RenderedOp tmp = JAI.create("scale", pb, null);
	          System.out.println(tmp.getWidth());
//			  mipmapLevelImage.createGraphics().drawRenderedImage(tmp, new AffineTransform());
//			  returnImage = mipmapLevelImage;

	     // Create the scale operation
			/*
			RenderedImage ri = mipmap.getImage(mipmapLevel);		
			if(rescaleMinMap) {
//				upscaleDisplayTransform= AffineTransform.getScaleInstance(2,2);
				if( img.getWidth() == (double) ri.getWidth()) {
					upscaleDisplayTransform= new AffineTransform();
				} else {
					double scale = (double) img.getWidth() / (double) ri.getWidth();
					System.out.println("mipmap size:" +ri.getWidth() + "x" + ri.getHeight());
					upscaleDisplayTransform= AffineTransform.getScaleInstance(scale, scale);
				}

//				System.out.println((double) img.getWidth() + "/ "+ (double) ri.getWidth()+"="+ (double) img.getWidth() / (double) ri.getWidth());
//				upscaleDisplayTransform.scale((double) img.getWidth() / (double) ri.getWidth(), (double) img.getHeight() / (double) ri.getHeight());
				rescaleMinMap = false;
			}
			
//			ri.getData().getPixel(x, y, iArray);
			mipmapLevelImage.createGraphics().drawRenderedImage(ri, upscaleDisplayTransform);
			returnImage = mipmapLevelImage;
			*/
			break;
			
		}


		return returnImage;
	}


	
//	protected ImageMIPMap getMipMap(RenderedImage img) { 
	//	return new ImageMIPMap(img,  downScaleTransform,  Interpolation.getInstance(Interpolation.INTERP_BILINEAR)); 
//	}
		

}
