package net.electroland.elvis.imaging;

import java.io.File;
import java.util.Vector;

import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.blobtracking.Tracker;
import net.electroland.elvis.imaging.imageFilters.BackgroundImage;
import net.electroland.elvis.imaging.imageFilters.BlobWriter;
import net.electroland.elvis.imaging.imageFilters.Blur;
import net.electroland.elvis.imaging.imageFilters.Copy;
import net.electroland.elvis.imaging.imageFilters.DetectContours;
import net.electroland.elvis.imaging.imageFilters.Dilate;
import net.electroland.elvis.imaging.imageFilters.Erode;
import net.electroland.elvis.imaging.imageFilters.Filter;
import net.electroland.elvis.imaging.imageFilters.ImageDifference;
import net.electroland.elvis.imaging.imageFilters.Mask;
import net.electroland.elvis.imaging.imageFilters.Scale;
import net.electroland.elvis.imaging.imageFilters.ThreshClamp;
import net.electroland.elvis.imaging.imageFilters.Unwarp;
import net.electroland.elvis.regions.GlobalRegionSnapshot;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class PresenceDetector extends ImageProcessor {

	public static final int CLAMP_VALUE = 20;

	public static enum ImgReturnType { RAW, MASK, UNWARP, BLUR, BACKGROUND,  DIFF, THRESH, DILATE, ERODE, CONTOUR, BLOBS, SCALE};

	ImgReturnType imgReturnType = ImgReturnType.RAW;

	int[][] threshMask = null;

	Filter[] filters;

	public ROI roi = null;
	public Mask mask;
	public Unwarp unwarp;
	public Copy raw;
	public ImageDifference diff;
	public BackgroundImage background;
	public DetectContours detectControus;
	public ThreshClamp thresh;
	public Blur blur;
	public Dilate dilate;
	public Erode erode;
	public BlobWriter blobWriter;
	public Scale scale;

	ImageConversion imageConversion = new ImageConversion();
	Vector<PolyRegion> regions = new Vector<PolyRegion>();

	CalcExtreema extreema;

	boolean calcExtreema = false;
	public boolean convertFromColor = false;

	public Tracker tracker = null;

	int srcWidth;
	int srcHeight;
	int roiWidth;
	int roiHeight;

	public Filter getCurrentFilter() {
		return filters[imgReturnType.ordinal()];

	}


	public int getWidth() {
		return roiWidth;
	}
	public int getHeight() {
		return roiHeight;
	}
	public int getSrcWidth() {
		return srcWidth;
	}
	public int getSrcHeight() {
		return srcHeight;
	}
	public PresenceDetector(ElProps props, boolean withTracker) {
		super();
		
		srcWidth = props.getProperty("srcWidth", 640);
		srcHeight = props.getProperty("srcHeight", 480);
		if(props.getProperty("useROI", true)) {
			roiWidth = props.getProperty("roiWidth", srcWidth);
			roiHeight = props.getProperty("roiHeight", srcHeight);
			roiWidth = (roiWidth >= srcWidth) ? srcWidth : roiWidth;
			roiHeight = (roiHeight >= srcHeight) ? srcHeight : roiHeight;
			if((roiWidth == srcWidth) && (roiHeight == srcHeight)) {
				// roi bigger or equal to src image so don't use
				roiWidth =srcWidth;
				roiHeight = srcHeight;
			} else {
				roi = new ROI(props);
			}
		} else {
			// no roi
			roiWidth =srcWidth;
			roiHeight = srcHeight;			
		}
		filters = new Filter[ImgReturnType.values().length];
		raw = new Copy();
//		mask = new Mask(props.getProperty("mask", "testMask.png"));
		mask = new Mask("testMask.png");
		extreema = new CalcExtreema();
		unwarp = new Unwarp(roiWidth,roiHeight, props);
		diff = new ImageDifference();
		blur = new Blur(5, props);
		background = new BackgroundImage(.001, 60, props);
		detectControus = new DetectContours(props);
		blobWriter = new BlobWriter(detectControus);
		dilate = new Dilate(1, props);
		erode = new Erode(1,props);
		thresh = new ThreshClamp(100, "", props);
		scale = new Scale(roiWidth,roiHeight, "grid", props);
		int i = 0;
		filters[i++] = raw;
		filters[i++] = mask;
		filters[i++] = unwarp;
		filters[i++] = blur;
		filters[i++] = background;
		filters[i++] = diff;
		filters[i++] = thresh;
		filters[i++] = dilate;
		filters[i++] = erode;
		filters[i++] = detectControus;
		filters[i++] = blobWriter;
		filters[i++] = scale;


		/*
		images = new IplImage[ImgReturnType.values().length];
		for(int i = 0; i < images.length; i++) {
			images[i] = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		}
		 */

		if(withTracker) {
			tracker = new Tracker(props);
			tracker.start();
		}
	}

	public static PresenceDetector createFromFile(ElProps props, File f) {
		GlobalRegionSnapshot grs = GlobalRegionSnapshot.load(f);
		PresenceDetector pd = new PresenceDetector(props, false);

		pd.background.adaptionParameter.setValue(grs.backgroundAdaptation);
		pd.thresh.parameters.get(0).setValue(grs.backgroundAdaptation);
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
		if(img==null) return null;
		if(roi != null) {
			roi.setForImage(img);
		}
		raw.apply(img);
		
		if(roi != null) {
			roi.removeForImage(img);
		}
		mask.apply(raw.getImage());
		unwarp.apply(mask.getImage());
		blur.apply(unwarp.getImage());
		if(background.apply(blur.getImage()) == null) return null;
		diff.apply(background.getImage(), blur.getImage());




		if(calcExtreema) {
			extreema.calc( diff.getImage(), null);
		}

		thresh.apply(diff.getImage());
		//		thresh2.apply(threshImage, threshImage2);

		dilate.apply(thresh.getImage());
		erode.apply(dilate.getImage());


		if(regions != null) {
			for(PolyRegion r : regions){
				if(r.isTriggered(erode.getImage())) {
				}
			}
		}

		detectControus.apply(erode.getImage());

		scale.apply(thresh.getImage());

		if(tracker != null) {
			detectControus.detectBlobs();
			tracker.queueBlobs(detectControus.getBlobs());

		}
		if(imgReturnType == ImgReturnType.BLOBS) {
			blobWriter.apply(erode.getImage());
		}


		
		return getCurrentFilter().getImage();



	}

	public Vector<Blob> getBlobs() {
		return detectControus.getBlobs();
	}

}
