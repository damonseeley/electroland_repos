package net.electroland.elvis.imaging;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;

import java.io.File;
import java.util.Vector;

import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.blobtracking.Tracker;
import net.electroland.elvis.regions.GlobalRegionSnapshot;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.cpp.opencv_core.IplImage;
public class PresenceDetector extends ImageProcessor {

	public static final int CLAMP_VALUE = 20;
	
	public static enum ImgReturnType { RAW, BLUR, BACKGROUND,  DIFF, THRESH, CONTOUR, BLOBS};

	ImgReturnType imgReturnType = ImgReturnType.RAW;

	int[][] threshMask = null;
	IplImage grayImage;
	IplImage diffImage;
	IplImage threshImage;
	IplImage contourImage; 
	IplImage blurImage; 
	
	BackgroundImage background;
	
	DetectContours detectControus;
	
	ThreshClamp thresh = new ThreshClamp(20);
	
	Blur blur;
	
	
	ImageConversion imageConversion = new ImageConversion();
	Vector<PolyRegion> regions = new Vector<PolyRegion>();

	CalcExtreema extreema;

	boolean calcExtreema = false;
	public boolean convertFromColor = false;
	
	public Tracker tracker = null;
	
	

	
	public PresenceDetector(ElProps props, int w, int h, boolean withTracker) {
		super(w, h);
		extreema = new CalcExtreema();
		blur = new Blur();
		background = new BackgroundImage(.001, 60);
		detectControus = new DetectContours(props);
		grayImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		blurImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		diffImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		threshImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		contourImage = IplImage.create(w, h, IPL_DEPTH_8U, 1);
		
		if(withTracker) {
			tracker = new Tracker(props);
			tracker.start();
		}
	}

	public static PresenceDetector createFromFile(ElProps props, File f) {
		GlobalRegionSnapshot grs = GlobalRegionSnapshot.load(f);
		PresenceDetector pd = new PresenceDetector(props, grs.w, grs.h, false);
		pd.setAdaptation(grs.backgroundAdaptation);
		pd.setThresh(grs.backgroundDiffThresh);
		pd.setRegions(grs.regions);
		return pd;
	}
	public void nextMode() {
		int nextOrd = imgReturnType.ordinal() + 1;
		if(nextOrd >= ImgReturnType.values().length) {
			nextOrd = 0;
		}
		imgReturnType =  ImgReturnType.values()[nextOrd];
		System.out.println("Displaying " + imgReturnType);
	}

	public void prevMode() {
		int prevOrd = imgReturnType.ordinal() - 1;
		if(prevOrd < 0) {
			prevOrd = ImgReturnType.values().length-1;
		}
		imgReturnType =  ImgReturnType.values()[prevOrd];
		System.out.println("Displaying " + imgReturnType);
	}

	public ImgReturnType getMode() { return imgReturnType; }

	
	public void setAdaptation(double d) {
		background.setAdaptation(d);
	}
	public double getAdaptation() {
		return background.getAdaptation();
	}
	public void setThresh(double d) {
		thresh.setThreshold(d);
		thresh.setClampValue(255);
	}
	
	public double getThresh() {
		return thresh.getThreshold();
	}
	
	public void setRegions(	Vector<PolyRegion> r) {
		regions = r;
	}
	
	public Vector<PolyRegion> getRegions(){
		return regions;
	}


	public float getMinBlobSize() {
		return detectControus.getMinBlobSize();
	}
	public float getMaxBlobSize() {
		return detectControus.getMaxBlobSize();
	}

	public void setMinBlobSize(float f) {
		 detectControus.setMinBlobsize(f);
	}
	public void setMaxBlobSize(float f) {
		 detectControus.setMaxBlobsize(f);
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
		// no longer supports color images
		
		blur.apply(grayImage, blurImage);
		IplImage bkImage = background.update(blurImage);
		if(bkImage == null) return null;

		ImageDifference.apply(bkImage, blurImage, diffImage);




		if(calcExtreema) {
			extreema.calc(diffImage, null);
		}

		thresh.apply(diffImage, threshImage);
//		thresh2.apply(threshImage, threshImage2);

		if(regions != null) {
			for(PolyRegion r : regions){
				if(r.isTriggered(threshImage)) {
				}
			}
		}
		
		detectControus.detectContours(threshImage , contourImage);
		if(tracker != null) {
			detectControus.detectBlobs();
			tracker.queueBlobs(detectControus.detectedBlobs);
			
		}
		
		IplImage returnImage = null;
		
		switch(imgReturnType) {
		case RAW:
			returnImage = img;
			break;
//		case GRAY: 
	//		returnImage = grayImage;
		//	break;
		case BLUR:
			returnImage = blurImage;
			break;
		case BACKGROUND: 
			returnImage = bkImage;
			break;
		case DIFF: 
			returnImage = diffImage;
			break; 
		case THRESH: 
			returnImage = threshImage;
			break;
		case BLOBS:
			detectControus.drawBlobs(contourImage);
			// no break on purpose
		case CONTOUR:
			returnImage = contourImage;
		}


		return returnImage;
	}

	public Vector<Blob> getBlobs() {
		return detectControus.detectedBlobs;
	}

}
