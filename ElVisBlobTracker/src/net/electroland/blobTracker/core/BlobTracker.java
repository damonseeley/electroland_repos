package net.electroland.blobTracker.core;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.blobDetection.Blob;
import net.electroland.blobDetection.Blobs;
import net.electroland.blobDetection.match.Tracker;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.blobTracker.util.RegionMap;
import net.electroland.elvis.imaging.BackgroundImage;
import net.electroland.elvis.imaging.ImageConversion;
import net.electroland.elvis.imaging.ImageDifference;
import net.electroland.elvis.imaging.ImageProcessor;
import net.electroland.elvis.imaging.ThreshClamp;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class BlobTracker extends ImageProcessor {
	public Tracker[] tracker;
	public RegionMap regionMap;
	
	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="axis";

	
	public boolean convertFromColor = false;


	ImageConversion imageConversion = new ImageConversion();
	IplImage grayImage;



	public static enum MODE { raw, background, diff, thresh, running };
	protected MODE mode = MODE.running;



	BackgroundImage background;
	IplImage diffImage;
	IplImage threshImage;
	IplImage threshImage2;

	ThreshClamp thresh = new ThreshClamp(2000);
	ThreshClamp thresh2 = new ThreshClamp(2000);

	ImageAcquirer srcStream;
	BufferedImage srcImage;

	Blobs blobs;
	
	public Vector<Blob> newFrameBlobs = new Vector<Blob>();
	
	public BlobTracker(int srcWidth, int srcHeight) {
		super(srcWidth, srcHeight);

		
		String mapFileName = ElProps.THE_PROPS.getProperty("regionMap","regionMap.png");
			regionMap = new RegionMap(mapFileName);

//			grayImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
			grayImage =IplImage.create(w, h, IPL_DEPTH_8U , 1);
//		scaledImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);





		diffImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		threshImage = IplImage.create(w, h, IPL_DEPTH_8U , 1);
		threshImage2 = IplImage.create(w, h, IPL_DEPTH_8U , 1);


		background = new BackgroundImage(.001, 15);

		blobs = new Blobs(srcWidth, srcHeight, regionMap);
		tracker = new Tracker[regionMap.size()];
		for(int i = 0; i < regionMap.size(); i++) {
			tracker[i]= new Tracker(ElProps.THE_PROPS, regionMap.getRegion(i));
			tracker[i].start();
		}
		
		

	}
	
	public void receiveErrorMsg(Exception cameraException) {
		// TODO Auto-generated method stub
		
	}




	public void resetBackground(int frameSkip) {
		if(background == null) return;
		background.reset(frameSkip);
		resetFPSCalc();
	}

	public void setBackgroundAdaptation(double d) {
		background.setAdaptation(d);
	}

	public double getAdaptation() {
		return background.getAdaptation();
	}





	
	public void nextMode() {
		int nextOrd = mode.ordinal() + 1;
		if(nextOrd >= MODE.values().length) {
			nextOrd = 0;
		}
		mode =  MODE.values()[nextOrd];
		System.out.println("Displaying " + mode);
	}

	public void prevMode() {
		int prevOrd = mode.ordinal() - 1;
		if(prevOrd < 0) {
			prevOrd = MODE.values().length-1;
		}
		mode =  MODE.values()[prevOrd];
		System.out.println("Displaying " + mode);
	}

	public MODE getMode() { return mode; }

	public void setThresh(double d) {
		if(d <= 0) return;
		if(d >= 65535) return;
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


	public void setSourceStream(String s) throws IOException {
		
		
		if(srcStream != null) {
			srcStream.stopRunning();
		}
		


//		int frameSkip = 2;
		if(s.equals(JMYRON_SRC)) {
//			frameSkip = 50;
			srcStream = new WebCam(w, h, 12, this, false);
		} else if(s.equals(LOCALAXIS_SRC)) {
			String ip = ElProps.THE_PROPS.getProperty("axisIP", "10.0.1.90");		
			String url = "http://" + ip + "/";
			String username = ElProps.THE_PROPS.getProperty("axisUsername", "root");
			String password = ElProps.THE_PROPS.getProperty("axisPassword", "n0h0");
			srcStream = new AxisCamera(url, w, h, 0, 0 , username, password, this);
		} else {
			srcStream = null;
			throw new IOException("Unknown source");
		}
		srcStream.start();
		// gridDetector.resetBackground(frameSkip);
	}

	public void setBackgroundImage(File f) throws IOException {
		if(srcStream != null) {
			srcStream.stopRunning();
			srcStream = null;
		}
		srcImage = ImageIO.read(f);
	}

	@Override
	public IplImage process(IplImage img) {

/*
		if(convertFromColor) {
			imageConversion.convertFromRGB(img, grayImage);			
		} else {
			imageConversion.convertFromGray(img, grayImage);
		}
*/

		IplImage bkImage = background.update(grayImage);
		if(bkImage == null) return null;


		ImageDifference.apply(bkImage, grayImage, diffImage);

		thresh.apply(diffImage, threshImage);
		thresh2.apply(threshImage, threshImage2);
		
		
		synchronized(blobs) { 
			Vector<Blob> allBlobs = new Vector<Blob>(newFrameBlobs.size());
			// probably should come up with some kind of double (or triple buffer) for this rather than re-allocating vectors every frame
			blobs.detectBlobs(threshImage2.getData());
			for(int i = 0; i < regionMap.size(); i++) {
				Vector<Blob> detected  = blobs.getDetectedBlobs(i);
				Vector<Blob> newBlobs = new Vector<Blob>(detected.size());
				newBlobs.addAll(detected);
				allBlobs.addAll(detected);
				tracker[i].queueBlobs(newBlobs);			
			}
			newFrameBlobs = allBlobs;
		}


		switch(mode) {
		case raw:
			return img;
		case background:
			return bkImage;
		case diff:
			return diffImage;
		case thresh:
		case running:
			return threshImage2;
		default:
			return img;

		}


	}

}
