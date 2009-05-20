package net.electroland.elvisVideoProcessor;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.concurrent.ArrayBlockingQueue;

import javax.media.jai.RenderedOp;

import net.electroland.elvis.imaging.BackgroundImage;
import net.electroland.elvis.imaging.ImageConversion;
import net.electroland.elvis.imaging.ImageDifference;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvisVideoProcessor.curveEditor.CurveEditor;
import net.electroland.elvisVideoProcessor.ui.CropConstructor;
import net.electroland.elvisVideoProcessor.ui.LAFaceFrame;
import net.electroland.elvisVideoProcessor.ui.LookupTable;
import net.electroland.elvisVideoProcessor.ui.MosaicConstructor;
import net.electroland.elvisVideoProcessor.ui.WarpGridConstructor;

public class LAFaceVideoProcessor extends Thread implements ImageReceiver{

	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="axis";


	int frameCnt = 0;
	long startTime;

	public boolean showGraphics = false;


	public boolean convertFromColor = false;

	ImageConversion imageConversion = new ImageConversion();


	BackgroundImage background;
	BufferedImage grayImage;
	BufferedImage diffImage;




	ImageAcquirer srcStream;
	BufferedImage srcImage;

	WarpGridConstructor warpGrid;
	LookupTable lookupTalbe;
	public CropConstructor crop;

	ElProps props;

	MosaicConstructor mosaic;


	RenderedOp cropOp;
	RenderedOp warpOp;
	RenderedOp lookupOp;
//	RenderedOp colorConvertOp;

//	AffineTransform cropTranslate;

	int[] lutCache = null;

	ArrayBlockingQueue<BufferedImage> queue = new ArrayBlockingQueue<BufferedImage>(2);


	public int w;
	public int h;



	public static enum MODE { raw, setWarp, viewWarp, crop, background, diff, colorAdjust, setMosiac, running };
	protected MODE mode = MODE.running;

	public LAFaceVideoProcessor(ElProps props) {
		this.props = props;
		showGraphics = ElProps.THE_PROPS.getProperty("showGraphics", false);
		w = props.getProperty("srcWidth", 640);
		h = props.getProperty("srcHeight", 480);

		String cropString = props.getProperty("crop","");
		if(cropString.length()==0) {
			crop = new CropConstructor(w,h,0,0,w,h);
		} else {
			crop = new CropConstructor(w,h,cropString);
		}


		/*
		warpedImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		grayImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		diffImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		 */
		background = new BackgroundImage(.5, 15);

		String warpGridStr =  props.getProperty("warpGrid", "");
		if(warpGridStr.length() == 0) {
			int warpGridWidth = props.getProperty("warpGridWidth", 6);
			int warpGridHeight = props.getProperty("warpGridHeight",3);			
			warpGrid = new WarpGridConstructor(warpGridWidth, warpGridHeight, w,h);
		} else {		
			warpGrid = new WarpGridConstructor(warpGridStr, w, h);
		}


		resetWarpAndROI();

//		if(! props.getProperty("showGraphics", true)) {
		String curveFile = props.getProperty("curveFile", "");
		System.out.println("loading " + curveFile);
		loadLutFile(curveFile);
		//	} else {
		//		resetLut(null);
		//	}

	}



	public void loadLutFile(String curveFile) {
		if(curveFile.length() == 0) {
			resetLut(null);
		} else {
			try {
				resetLut(CurveEditor.constructLutFromFile(65536, curveFile));
			} catch (IOException e) {
				e.printStackTrace();
				resetLut(null);
			}
		}		
	}

	public void resetLut(int[] lut) {
		if(lut == null) {
			lookupOp  = new LookupTable(warpOp).getLookupOp();			
		} else {
			lookupOp  = new LookupTable(warpOp, lut).getLookupOp();
		}

	}


	public void resetWarpAndROI() {
		System.out.println("resetting warp");

//		cropOp = crop.getCropOp();  UNDONE shoud be first


		grayImage = new BufferedImage(crop.rect.width, crop.rect.height, BufferedImage.TYPE_USHORT_GRAY);
		diffImage = new BufferedImage(crop.rect.width, crop.rect.height, BufferedImage.TYPE_USHORT_GRAY);




		warpGrid.reset();
		// should work with cropped image but doesn't UNDONE
//		warpGrid.setSrcDims(crop.rect.width, crop.rect.height);
		warpGrid.setSrcDims(w, h);

		//UNDONE crop first
//		warpOp = warpGrid.getWarpOp(cropOp);
		warpOp = warpGrid.getWarpOp(new BufferedImage(w,h, BufferedImage.TYPE_BYTE_GRAY));

		cropOp = crop.getCropOp(warpOp);

//		warpOp = crop


		String mosaicString = props.getProperty("mosaicRects", "");
		int mosaicWidth = props.getProperty("mosaicWidth", crop.rect.width);
		int mosaicHeight = props.getProperty("mosaicHeight", 21);
		int mosaicSubDivs = props.getProperty("mosaicSubDivs", 7);
		if(mosaicString.length() == 0) {
			mosaic = new MosaicConstructor(crop.rect.width, crop.rect.height,  mosaicWidth, mosaicHeight, mosaicSubDivs, 2);			
		} else {
			mosaic = new MosaicConstructor(crop.rect.width, crop.rect.height,mosaicWidth, mosaicHeight, mosaicSubDivs, mosaicString);

		}



//		roi = roiConstructor.getRoi();

		resetBackground(background.getRemainingFrameSkip()+2);

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

	public void setSourceStream(String s) throws IOException {


		if(srcStream != null) {
			srcStream.stopRunning();
		}



//		int frameSkip = 2;
		if(s.equals(JMYRON_SRC)) {
//			frameSkip = 50;
			srcStream = new WebCam(w, h, 15, this, false);
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






	public BufferedImage process(BufferedImage img) {
//		System.out.println("process");
		if(lutCache != null) {
			resetLut(lutCache);
			lutCache = null;
		}


		// SHOULD CROP FIRST UNDONE
		if((mode == MODE.raw) ||(mode == MODE.setWarp)) {
			return img;
		}


		warpOp.setSource(img, 0);

//		cropOp.setSource(img, 0);
//		warpOp.setSource(cropOp.getAsBufferedImage(), 0);
//		BufferedImage warped = warpOp.getAsBufferedImage();
		cropOp.setSource(warpOp.getAsBufferedImage(), 0);
		BufferedImage warped = cropOp.getAsBufferedImage();

		if((warped.getWidth() != grayImage.getWidth()) || (warped.getHeight() != grayImage.getHeight())) {
			grayImage = new BufferedImage(warped.getWidth(), warped.getHeight(), BufferedImage.TYPE_USHORT_GRAY);
		}
		try {
			if(convertFromColor) {
				imageConversion.convertFromRGB(cropOp.getAsBufferedImage(), grayImage);			
			} else {
				imageConversion.convertFromGray(cropOp.getAsBufferedImage(), grayImage);
			}
		} catch (RuntimeException e) {
			e.printStackTrace();
			System.out.println(grayImage.getWidth());
			System.out.println(cropOp.getAsBufferedImage().getWidth());
			System.out.println(grayImage.getHeight());
			System.out.println(cropOp.getAsBufferedImage().getHeight());
			return img;
		}

		BufferedImage bkImage = background.update(grayImage);
		if(bkImage == null) return warpOp.getAsBufferedImage();

		ImageDifference.apply(bkImage, grayImage, diffImage);

		if(lookupOp == null) {
			System.out.println("lookupOp is null");
			return img;
		}
		lookupOp.setSource(grayImage, 0);
		lookupOp.getNewRendering();

		switch(mode) {
		case raw:
		case crop:
			//UNDONE
//			return img;
			return warpOp.getAsBufferedImage();
		case setWarp:
//			return cropOp.getAsBufferedImage();
			return img;
		case viewWarp:
			return warpOp.getAsBufferedImage();
		case background:
			return bkImage;
		case diff:
			return diffImage;
		case colorAdjust:
			return lookupOp.getAsBufferedImage();
		case setMosiac: {
			return lookupOp.getAsBufferedImage();
		}
		case running:
		default: {
			if(showGraphics) {
				BufferedImage bi = lookupOp.getAsBufferedImage();
				int y = 0;
				BufferedImage[] imgs;
				try {
					mosaic.processImage(lookupOp);
					imgs = mosaic.getImage();
					for(BufferedImage curImg : imgs) {
						bi.createGraphics().drawImage(curImg, 0,y, null);
						y+= curImg.getHeight();
					}
					return bi;

				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					return img;
				}
			} else {
				mosaic.processImage(lookupOp);				
				return img; // not really used
			}
		}


		}




	}
	
	public BufferedImage[] getMosaics() throws InterruptedException {
		return mosaic.getImage();
	}

	public MODE getMode() {
		return mode;
	}

	public void nextMode() {
		mode = MODE.values()[(mode.ordinal() + 1)%MODE.values().length];
	}
	public void prevMode() {
		int prev = (mode.ordinal() -1);
		prev = (prev < 0) ? MODE.values().length -1:prev;
		mode = MODE.values()[prev];
	}

	public static void main(String arg[]) throws IOException {
		if(arg.length > 0) {
			ElProps.init(arg[0]);
		} else {
			ElProps.init("LAFace.props");
		}

		LAFaceVideoProcessor lafvp = new LAFaceVideoProcessor(ElProps.THE_PROPS);

		if(ElProps.THE_PROPS.getProperty("showGraphics", true)) {
			new LAFaceFrame("LAFace", lafvp, ElProps.THE_PROPS);
		}

		lafvp.setBackgroundAdaptation(ElProps.THE_PROPS.setProperty("adaptation", .1));

		try {

//			lafvp.setSourceStream(ElProps.THE_PROPS.getProperty("camera", JMYRON_SRC));
			lafvp.setSourceStream(ElProps.THE_PROPS.getProperty("camera", LOCALAXIS_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}


		lafvp.start();



	}
	public void setLutCache(int[] lut) {
		lutCache = lut;
	}

	public WarpGridConstructor getROIConstructor() {
		return warpGrid;
	}
	public MosaicConstructor getMosaicConstructor() {
		return mosaic;
	}


	public void resetFPSCalc() {
		frameCnt = 0;
		startTime = System.currentTimeMillis();
	}


	public float getFPS() {
		return (1000.0f * frameCnt) / ((float) (System.currentTimeMillis() - startTime));
	}


	public void addImage(BufferedImage i) {
		try {
			queue.put(i);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	boolean isRunning = true;

	public void stopRunning() {
		isRunning = false;
		try {
			queue.put(new BufferedImage(crop.rect.width, crop.rect.height, BufferedImage.TYPE_USHORT_GRAY));
		} catch (InterruptedException e) {
		}
	}

	BufferedImage img;

	public BufferedImage getImage() { 
		return img;
	}

	public void run() {
		startTime = System.currentTimeMillis();

		while(isRunning) {
			try {
				img = process(queue.take());
			} catch (InterruptedException e) {
				if(isRunning) {
					e.printStackTrace();
				}
			}

		}
	}


}
