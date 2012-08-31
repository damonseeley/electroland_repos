package net.electroland.elvis.imaging;

import java.io.File;
import java.net.SocketException;
import java.net.UnknownHostException;
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
import net.electroland.elvis.imaging.imageFilters.GridUnwarp;
import net.electroland.elvis.imaging.imageFilters.ImageDifference;
import net.electroland.elvis.imaging.imageFilters.Mask;
import net.electroland.elvis.imaging.imageFilters.Scale;
import net.electroland.elvis.imaging.imageFilters.ThreshClamp;
import net.electroland.elvis.imaging.imageFilters.Unwarp;
import net.electroland.elvis.manager.GlobalSettingsPanelMig;
import net.electroland.elvis.net.PresenceGridUPDBroadcaster;
import net.electroland.elvis.net.RegionUPDBroadcaster;
import net.electroland.elvis.net.TrackUPDBroadcaster;
import net.electroland.elvis.regions.GlobalRegionSnapshot;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.cpp.opencv_core.IplImage;




public class PresenceDetector extends ImageProcessor {


	public static enum ImgReturnType { RAW, BLUR, MASK, UNWARP, GRID_UNWARP, BACKGROUND,  DIFF, THRESH, DILATE, ERODE, CONTOUR, BLOBS, GRIDCOUNT};


	boolean netImageNeedsUpdate = false;
	IplImage netImage;

	ImgReturnType imgReturnType = ImgReturnType.RAW;
	ImgReturnType netImgReturnType = ImgReturnType.RAW;

	int[][] threshMask = null;

	Filter[] filters;

	public ROI roi = null;
	public Mask mask;
	public Unwarp unwarp;
	public GridUnwarp gridUnwarp;
	public Copy raw;
	public ImageDifference diff;
	public BackgroundImage background;
	public DetectContours detectControus;
	public ThreshClamp thresh;
	public Blur blur;
	public Dilate dilate;
	public Erode erode;
	public BlobWriter blobWriter;
	///public GridCount gridcount;
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

	TrackUPDBroadcaster trackbroadcaster;
	RegionUPDBroadcaster regionBroadcaster;
	PresenceGridUPDBroadcaster gridBroadcaster;

	public Filter getCurrentNetFilter() {
		return filters[netImgReturnType.ordinal()];
	}
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
	public PresenceDetector(ElProps props) throws SocketException, UnknownHostException {
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
		gridUnwarp = new GridUnwarp(roiWidth, roiHeight, props);
		diff = new ImageDifference();
		blur = new Blur(5, props);
		background = new BackgroundImage(.001, 60, props);
		detectControus = new DetectContours(props);
		blobWriter = new BlobWriter(detectControus);
		dilate = new Dilate(1, props);
		erode = new Erode(1,props);
		thresh = new ThreshClamp(100, "", props);
		scale = new Scale(roiWidth,roiHeight, "grid", props);
		//		gridcount = new GridCount(roiWidth, roiHeight, props);
		int i = 0;
		filters[i++] = raw;
		filters[i++] = blur;
		filters[i++] = mask;
		filters[i++] = unwarp;
		filters[i++] = gridUnwarp;
		filters[i++] = background;
		filters[i++] = diff;
		filters[i++] = thresh;
		filters[i++] = dilate;
		filters[i++] = erode;
		filters[i++] = detectControus;
		filters[i++] = blobWriter;
		filters[i++] = scale;
		//		filters[i++] = gridcount;

		File regionFile = new File(props.getProperty("regionFile", "testRegions.elv"));
		if(regionFile.exists()) {
			setRegions(GlobalSettingsPanelMig.load(regionFile));			
		}


		/*
		images = new IplImage[ImgReturnType.values().length];
		for(int i = 0; i < images.length; i++) {
			images[i] = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		}
		 */

		if(props.getProperty("trackerIsOn", true)) {
			tracker = new Tracker(props);
			tracker.start();


			if(props.getProperty("broadcastTracks", true)) {
				String address = props.getProperty("broadcastTracksAddress", "localhost");
				int port = props.getProperty("broadcastTracksPort", 3456);
				trackbroadcaster = new TrackUPDBroadcaster(address, port);
				tracker.addListener(trackbroadcaster);
				trackbroadcaster.start();
			}
		}
		if(props.getProperty("broadcastRegions", true)) {
			String address = props.getProperty("broadcastRegionsAddress", "localhost");
			int port = props.getProperty("broadcastRegionsPort", 3457);
			regionBroadcaster = new RegionUPDBroadcaster(address, port);
			regionBroadcaster.start();

		}
		if(props.getProperty("broadcastGrid", true)) {
			String address = props.getProperty("broadcastGridAddress", "localhost");
			int port = props.getProperty("broadcastGridPort", 3458);
			gridBroadcaster = new PresenceGridUPDBroadcaster(address, port);	
			gridBroadcaster.start();
		}



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


	public void setNetImageReturn(String s) {
		//	System.out.println("getting value of :"+ s+":");
		//		netImgReturnType = ImgReturnType.valueOf(s.toLowerCase());
		//		netImgReturnType = ImgReturnType.RAW;
		netImgReturnType = ImgReturnType.valueOf(s);
	}

	public void setNetImageReturn(ImgReturnType ret) {
		netImgReturnType = ret;
	}

	public void setImageReturn(ImgReturnType ret) {
		imgReturnType = ret;
	}

	public IplImage getNetImage() {
		this.netImageNeedsUpdate = true;
		return netImage;
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
		blur.apply(raw.getImage());
		mask.apply(blur.getImage());
		unwarp.apply(mask.getImage());
		gridUnwarp.apply(unwarp.getImage());
		if(background.apply(gridUnwarp.getImage()) == null) return null;
		diff.apply(background.getImage(), gridUnwarp.getImage());




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

		//		gridcount.apply(thresh.getImage());
		scale.apply(erode.getImage());

		if(tracker != null) {
			detectControus.detectBlobs();
			tracker.queueBlobs(detectControus.getBlobs());

		}
		if(imgReturnType == ImgReturnType.BLOBS) {
			blobWriter.apply(erode.getImage());
		}

		if(netImageNeedsUpdate) {
			netImage =  getCurrentNetFilter().getImage().clone();
			netImageNeedsUpdate = false;
		}

		//		TrackUPDBroadcaster trackbroadcaster;
		//		RegionUPDBroadcaster regionBroadcaster;
		//		PresenceGridUPDBroadcaster gridBroadcaster;
		regionBroadcaster.updateRegions(regions);
		gridBroadcaster.updateGrid(scale.getSmallImg());

		return getCurrentFilter().getImage();



	}

	public Vector<Blob> getBlobs() {
		return detectControus.getBlobs();
	}

}
