package net.electroland.elvisVideoProcessor;

import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.media.jai.RenderedOp;
import javax.media.jai.operator.CropDescriptor;

import net.electroland.elvis.imaging.BackgroundImage;
import net.electroland.elvis.imaging.ImageConversion;
import net.electroland.elvis.imaging.ImageDifference;
import net.electroland.elvis.imaging.ImageProcessor;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvisVideoProcessor.curveEditor.CurveEditor;
import net.electroland.elvisVideoProcessor.ui.LAFaceFrame;
import net.electroland.elvisVideoProcessor.ui.LookupTable;
import net.electroland.elvisVideoProcessor.ui.WarpGridConstructor;

public class LAFaceVideoProcessor extends ImageProcessor {

	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="axis";

	public boolean convertFromColor = false;

	ImageConversion imageConversion = new ImageConversion();


	BackgroundImage background;
	BufferedImage grayImage;
	BufferedImage diffImage;




	ImageAcquirer srcStream;
	BufferedImage srcImage;

	WarpGridConstructor warpGrid;
	LookupTable lookupTalbe;

	ElProps props;


	RenderedOp cropOp;
	RenderedOp warpOp;
	RenderedOp lookupOp;
//	RenderedOp colorConvertOp;

	AffineTransform cropTranslate;

	int[] lutCache = null;





	public static enum MODE { raw, setWarp, viewWarp, background, diff, colorAdjust,running };
	protected MODE mode = MODE.running;

	public LAFaceVideoProcessor(ElProps props) {
		super(props.getProperty("srcWidth", 640), props.getProperty("srcHeight", 480));
		this.props = props;
		


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
		warpGrid.reset();


		cropOp = CropDescriptor.create(new BufferedImage(w,h, BufferedImage.TYPE_USHORT_GRAY) , 
				(float)warpGrid.getCropX(), (float) warpGrid.getCropY(), (float)warpGrid.getCropW(), (float)warpGrid.getCropH(), null);


		cropTranslate = new AffineTransform();
		cropTranslate.translate(- warpGrid.getCropX(),- warpGrid.getCropY());

		grayImage = new BufferedImage(warpGrid.getCropW(),warpGrid.getCropH(), BufferedImage.TYPE_USHORT_GRAY);
		diffImage = new BufferedImage(warpGrid.getCropW(),warpGrid.getCropH(), BufferedImage.TYPE_USHORT_GRAY);

		warpOp = warpGrid.getWarpOp(cropOp);



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
	
	
	



	@Override
	public BufferedImage process(BufferedImage img) {
//		System.out.println("process");
		if(lutCache != null) {
			resetLut(lutCache);
			lutCache = null;
		}

		if((mode == MODE.raw) ||(mode == MODE.setWarp)) {
			return img;
		}
		

		cropOp.setSource(img, 0);
		warpOp.setSource(cropOp.getAsBufferedImage(), 0);
		
		BufferedImage warped = warpOp.getAsBufferedImage();
		if((warped.getWidth() != grayImage.getWidth()) || (warped.getHeight() != grayImage.getHeight())) {
			grayImage = new BufferedImage(warped.getWidth(), warped.getHeight(), BufferedImage.TYPE_USHORT_GRAY);
		}
		try {
			if(convertFromColor) {
				imageConversion.convertFromRGB(warpOp.getAsBufferedImage(), grayImage);			
			} else {
				imageConversion.convertFromGray(warpOp.getAsBufferedImage(), grayImage);
			}
		} catch (RuntimeException e) {
			System.out.println(grayImage.getWidth());
			System.out.println(warpOp.getAsBufferedImage().getWidth());
			System.out.println(grayImage.getHeight());
			System.out.println(warpOp.getAsBufferedImage().getHeight());
			throw e;
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
		case setWarp:
			return img;
		case viewWarp:
			return warpOp.getAsBufferedImage();
		case background:
			return bkImage;
		case diff:
			return diffImage;
		case colorAdjust:
			return lookupOp.getAsBufferedImage();
		case running:
		default:
			return lookupOp.getAsBufferedImage();

		}

		//	grayImage.createGraphics().drawLine(0, 0, warpGrid.getCropW(), warpGrid.getCropH());


		//	warpOp.getNewRendering();

//		return bkImage;
		/*




		warpOp.setSource(img, 0);
//		lookupOp.setSource(img, 0);

		warpedImage.createGraphics().drawRenderedImage(lookupOp,new AffineTransform());



		BufferedImage bkImage = background.update(grayImage);
		if(bkImage == null) return warpedImage;


		ImageDifference.apply(bkImage, grayImage, diffImage);



		switch(mode) {
		case raw:
		case setWarp:
			return img;
		case viewWarp:
			return warpedImage;
		case background:
			return bkImage;
		case diff:
			return diffImage;
		case running:
			return diffImage;
		default:
			return img;

		}
		 */


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

}
