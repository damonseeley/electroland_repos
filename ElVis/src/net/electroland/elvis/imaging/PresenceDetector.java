package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;

import java.io.File;
import java.util.Vector;

import net.electroland.elvis.regions.GlobalRegionSnapshot;
import net.electroland.elvis.regions.PolyRegion;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
public class PresenceDetector extends ImageProcessor {

	public static final int CLAMP_VALUE = 65535;
	
	public static enum ImgReturnType { RAW, GRAY, BGRND, DIFF, THRESH};

	ImgReturnType imgReturnType = ImgReturnType.RAW;

	int[][] threshMask = null;
	IplImage grayImage;
	IplImage diffImage;
	IplImage threshImage;
	IplImage threshImage2;
	BackgroundImage background;
	
	ThreshClamp thresh = new ThreshClamp(2000);
	ThreshClamp thresh2 = new ThreshClamp(2000);
	
	
	ImageConversion imageConversion = new ImageConversion();
	Vector<PolyRegion> regions = new Vector<PolyRegion>();

	CalcExtreema extreema;

	boolean calcExtreema = false;
	public boolean convertFromColor = false;

	
	public PresenceDetector(int w, int h) {
		super(w, h);
		extreema = new CalcExtreema();
		background = new BackgroundImage(.001, 60);
		grayImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		diffImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		threshImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		threshImage2 = IplImage.create(w, h, IPL_DEPTH_8U , 1);
	}

	public static PresenceDetector createFromFile(File f) {
		GlobalRegionSnapshot grs = GlobalRegionSnapshot.load(f);
		PresenceDetector pd = new PresenceDetector(grs.w, grs.h);
		pd.setAdaptation(grs.backgroundAdaptation);
		pd.setThresh(grs.backgroundDiffThresh);
		pd.setRegions(grs.regions);
		return pd;
	}

	
	public void setAdaptation(double d) {
		background.setAdaptation(d);
	}
	public double getAdaptation() {
		return background.getAdaptation();
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
	
	public void setRegions(	Vector<PolyRegion> r) {
		regions = r;
	}
	
	public Vector<PolyRegion> getRegions(){
		return regions;
	}






	public void resetBackground(int frameSkip) {
		background.reset(frameSkip);
		resetFPSCalc();
	}


	public void recalcExtreema() {
		calcExtreema = true;
	}

	public double getMin() {
		return extreema.getMin();
	}

	public double getMax() {
		return extreema.getMax();
	}

	
	public void setImageReturn(ImgReturnType ret) {
		imgReturnType = ret;
	}

	public IplImage process(IplImage img) {

		grayImage = img;
		/*  TODO: handel color correction 
		if(convertFromColor) {
			imageConversion.convertFromRGB(img, grayImage);			
		} else {
			imageConversion.convertFromGray(img, grayImage);
		}
		*/
		IplImage bkImage = background.update(grayImage);
		if(bkImage == null) return null;

		ImageDifference.apply(bkImage, grayImage, diffImage);




		if(calcExtreema) {
			extreema.calc(diffImage, null);
		}

		thresh.apply(diffImage, threshImage);
		thresh2.apply(threshImage, threshImage2);

		if(regions != null) {
			for(PolyRegion r : regions){
				if(r.isTriggered(threshImage2)) {
				}
			}
		}
		
		
		
		IplImage returnImage = null;
		
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
		}


		return returnImage;
	}


	

}
