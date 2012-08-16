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

	public static enum ImgReturnType { RAW, UNWARP, BLUR, BACKGROUND,  DIFF, THRESH, DILATE, ERODE, CONTOUR, BLOBS, SCALE};

	ImgReturnType imgReturnType = ImgReturnType.RAW;

	int[][] threshMask = null;

	Filter[] filters;

	public Unwarp unwarp;
	public NoOpFilter raw;
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


	public Filter getCurrentFilter() {
		return filters[imgReturnType.ordinal()];

	}


	public PresenceDetector(ElProps props, int w, int h, boolean withTracker) {
		super(w, h);
		filters = new Filter[ImgReturnType.values().length];
		raw = new NoOpFilter();
		extreema = new CalcExtreema();
		unwarp = new Unwarp(w,h, props);
		diff = new ImageDifference();
		blur = new Blur(5, props);
		background = new BackgroundImage(.001, 60, props);
		detectControus = new DetectContours(props);
		blobWriter = new BlobWriter(detectControus);
		dilate = new Dilate(1, props);
		erode = new Erode(1,props);
		thresh = new ThreshClamp(100, "", props);
		scale = new Scale(w,h, "grid", props);
		filters[0] = raw;
		filters[1] = unwarp;
		filters[2] = blur;
		filters[3] = background;
		filters[4] = diff;
		filters[5] = thresh;
		filters[6] = dilate;
		filters[7] = erode;
		filters[8] = detectControus;
		filters[9] = blobWriter;
		filters[10] = scale;


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
		PresenceDetector pd = new PresenceDetector(props, grs.w, grs.h, false);

		pd.background.adaptionParameter.setValue(grs.backgroundAdaptation);
		pd.thresh.parameters.get(0).setValue(grs.backgroundAdaptation);
		//		pd.setAdaptation(grs.backgroundAdaptation);
		//		pd.setThresh(grs.backgroundDiffThresh);
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

	/*
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
	 */
	public void setRegions(	Vector<PolyRegion> r) {
		regions = r;
	}


	public Vector<PolyRegion> getRegions(){
		return regions;
	}

/*
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
*/

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
		raw.apply(img);
		unwarp.apply(raw.getImage());
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
			tracker.queueBlobs(detectControus.detectedBlobs);

		}
		if(imgReturnType == ImgReturnType.BLOBS) {
			blobWriter.apply(erode.getImage());
		}

		
		return getCurrentFilter().getImage();
		
		

	}

	public Vector<Blob> getBlobs() {
		return detectControus.detectedBlobs;
	}

}
