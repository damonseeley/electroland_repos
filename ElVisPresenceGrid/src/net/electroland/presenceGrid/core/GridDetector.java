package net.electroland.presenceGrid.core;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.PerspectiveTransform;
import javax.media.jai.RenderedOp;
import javax.media.jai.WarpGrid;

import net.electroland.elvis.imaging.BackgroundImage;
import net.electroland.elvis.imaging.ImageConversion;
import net.electroland.elvis.imaging.ImageDifference;
import net.electroland.elvis.imaging.ImageProcessor;
import net.electroland.elvis.imaging.ThreshClamp;
import net.electroland.presenceGrid.util.GridProps;

public class GridDetector extends ImageProcessor {

	public boolean convertFromColor = false;


	protected int srcWidth;
	protected int srcHeight;

//	protected int xOffset;
//	protected int yOffset;


//	protected int cropWidth;
//	protected int cropHeight;


	protected int outputWidth;
	protected int outputHeight;



//	RenderedOp cropOp;
	RenderedOp scaleOp;


	RenderedOp warpOp;

	ImageConversion imageConversion = new ImageConversion();
	BufferedImage grayImage;


	float transformWidth;
	float transformHeight;


	public static enum MODE { raw, scaled, background, diff, thresh, running, setCrop };
	protected MODE mode = MODE.running;


	public GridListener theListener; //TODO remove the public static its a hack because I'm lazy

	BackgroundImage background;
	BufferedImage scaledImage;
	BufferedImage diffImage;
	BufferedImage threshImage;
	BufferedImage threshImage2;

	ThreshClamp thresh = new ThreshClamp(2000);
	ThreshClamp thresh2 = new ThreshClamp(2000);

	public Point.Float[][] origTransformPoints ;
	public Point.Float[][] mappedTransformPoints ;

	
	// added by DS 2012 for comaptibility with later ElVis build
	public void receiveErrorMsg(Exception e) {
		System.out.println(e);
	}

	public GridDetector(int srcWidth, int srcHeight, int outputWidth, int outputHeight) {
		super(outputWidth, outputHeight);
		this.srcWidth = srcWidth;
		this.srcHeight = srcHeight;


		this.outputWidth = outputWidth;
		this.outputHeight = outputHeight;

		grayImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		scaledImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);





		diffImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		threshImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);
		threshImage2 = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);

		/*
		background = new BackgroundImage(.001, 60);
	//	mipmapLevelImage = new BufferedImage(w,h,BufferedImage.TYPE_USHORT_GRAY);

//		downScaleTransform= AffineTransform.getScaleInstance(.5, .5);
		 */

		background = new BackgroundImage(.001, 15);


	}

	public Point.Float getSelectedPoint(int x, int y, int r) {
		float dist = r;
		Point.Float curPoint = null;

		for(int i = 0; i < mappedTransformPoints.length; i++) {
			for(int j = 0; j < mappedTransformPoints[0].length; j++) {
				Point.Float p = mappedTransformPoints[i][j];
				float tmp = p.x - x;
				tmp *= tmp;
				float newDist = tmp;
				tmp = p.y - y;
				tmp *= tmp;
				newDist += tmp;
				if(newDist < dist ) {
					curPoint = p;
				}

			}

		}
		return curPoint;
		/*			
		for(Point p : transformPoints) {
			float tmp = p.x - x;
			tmp *= tmp;
			float newDist = tmp;
			tmp = p.y - y;
			tmp *= tmp;
			newDist += tmp;
			if(newDist < dist ) {
				curPoint = p;
			}
		}
		return curPoint;
		 */

	}
	/*
	public Point getSelectedPoint(int x, int y, int r) {
		float dist = r;
		Point curPoint = null;
		for(Point p : transformPoints) {
			float tmp = p.x - x;
			tmp *= tmp;
			float newDist = tmp;
			tmp = p.y - y;
			tmp *= tmp;
			newDist += tmp;
			if(newDist < dist ) {
				curPoint = p;
			}
		}
		return curPoint;


	}
	 */


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

	public void createSetWarpGrid() {
		Point transformCells = GridProps.getGridProps().getProperty("transformDims", new Point(3,2));
		String cellMapStr = GridProps.getGridProps().getProperty("transformMapping", "");

		String[] cellMapStrArg  = cellMapStr.split(",");

		if(cellMapStrArg.length != 2 * (transformCells.x + 1) * (transformCells.y + 1)) {
			System.out.println("Reconstructing transfrom grid from scratch only " + cellMapStrArg.length + " element in prop");
			resetWarpGrid(transformCells.x, transformCells.y);
		} else {
			System.out.println("reading tranform grid from props");
			Point.Float[][] map = new Point.Float[transformCells.x + 1][transformCells.y + 1];
			int curS = 0;
			for(int i = 0; i < transformCells.x + 1 ; i++) {
				for(int j = 0; j < transformCells.y + 1 ; j++) {
					float x = Float.parseFloat(cellMapStrArg[curS++]);
					float y = Float.parseFloat(cellMapStrArg[curS++]);
					map[i][j] = new Point.Float(x,y);
				}

			}
			resetWarpGrid(map);
		}
	}

	public void resetWarpGrid(Point.Float[][] mappedTransformPoints ) {
		this.mappedTransformPoints = mappedTransformPoints;

		int xCnt = mappedTransformPoints.length -1;
		int yCnt = mappedTransformPoints[0].length -1;

		System.out.println("src="+srcWidth + "x" + srcHeight);
		//int cellW = (int) Math.floor(((float) srcWidth /(float) (xCnt)));
		//int cellH = (int) Math.floor(((float) srcHeight /(float) (yCnt )));

		float cellW =  ((float) srcWidth /(float) (xCnt));
		float cellH =  ((float) srcHeight /(float) (yCnt ));

		transformWidth = cellW;
		transformHeight = cellH;

		System.out.println("TransformCellSize = " + cellW + "x" + cellH);

		origTransformPoints = new Point.Float[xCnt+1][yCnt+1];


		float x =0;
		float y = 0;

		for(int i = 0; i < xCnt +1; i++) {
			Point.Float[] rowO = origTransformPoints[i];
			for(int j = 0; j < yCnt+1; j++) {
				rowO[j] = new Point.Float(x,y);
				y += cellH;				
			}
			x += cellW;
			y = 0;
		}


		resetCropAndScale();


	}

	public void resetWarpGrid(int xCnt, int yCnt) {

		System.out.println("src="+srcWidth + "x" + srcHeight);

		//int cellW = (int) Math.floor(((float) srcWidth /(float) (xCnt)));
		//int cellH = (int) Math.floor(((float) srcHeight /(float) (yCnt )));

		float cellW =  ((float) srcWidth /(float) (xCnt));
		float cellH =  ((float) srcHeight /(float) (yCnt ));

		transformWidth = cellW;
		transformHeight = cellH;

		System.out.println("TransformCellSize = " + cellW + "x" + cellH);

		origTransformPoints = new Point.Float[xCnt+1][yCnt+1];
		mappedTransformPoints = new Point.Float[xCnt+1][yCnt+1];


		float x =0;
		float y = 0;

		for(int i = 0; i < xCnt +1; i++) {
			Point.Float[] rowO = origTransformPoints[i];
			Point.Float[] rowM = mappedTransformPoints[i];
			for(int j = 0; j < yCnt+1; j++) {
				rowO[j] = new Point.Float(x,y);
				rowM[j] = new Point.Float(x,y);
				y += cellH;				
			}
			x += cellW;
			y = 0;
		}



		resetCropAndScale();

	}


	public void resetCropAndScale() {






//		System.out.println("0->" + transformPoints.get(0).x + " " + transformPoints.get(0).y);

		/*
		PerspectiveTransform pt = PerspectiveTransform.getQuadToSquare(
			transformPoints.get(0).x, transformPoints.get(0).y,
			transformPoints.get(1).x, transformPoints.get(1).y,
			transformPoints.get(2).x, transformPoints.get(2).y,
			transformPoints.get(3).x, transformPoints.get(3).y);
		 */


		int curPt = 0;
		PerspectiveTransform[][] transforms = new PerspectiveTransform[(origTransformPoints.length -1)][(origTransformPoints[0].length -1)];
		for(int i = 0; i < origTransformPoints.length-1; i++) {
			for(int j = 0; j < origTransformPoints[0].length-1 ; j++) {
				transforms[i][j] = PerspectiveTransform.getQuadToQuad(
						origTransformPoints[i][j].x, origTransformPoints[i][j].y,
						origTransformPoints[i][j+1].x, origTransformPoints[i][j+1].y,
						origTransformPoints[i+1][j].x, origTransformPoints[i+1][j].y,
						origTransformPoints[i+1][j+1].x, origTransformPoints[i+1][j+1].y,

						mappedTransformPoints[i][j].x, mappedTransformPoints[i][j].y,
						mappedTransformPoints[i][j+1].x, mappedTransformPoints[i][j+1].y,
						mappedTransformPoints[i+1][j].x, mappedTransformPoints[i+1][j].y,
						mappedTransformPoints[i+1][j+1].x, mappedTransformPoints[i+1][j+1].y

				);
				curPt++;
			}
		}

		float[] transformPositions = new float[(srcWidth) * (srcHeight) * 2];

		Point2D.Float origPt = new Point2D.Float() ;
		Point2D.Float tranformedPt = new Point2D.Float() ;

		int i = 0;


		for(float y = 0f; y< srcHeight;y++) {
			int transformJ = (int) Math.floor(((float) y) / ((float) transformHeight));
			for(float x = 0f; x< srcWidth;x++) {
				int transformI = (int) Math.floor(((float) x) / ((float) transformWidth));

				origPt.setLocation(x,y);
				transforms[transformI][transformJ].transform(origPt, tranformedPt);
				//	System.out.println(i + ":"  + x + "," + y + " --> " + tranformedPt.x + "," + tranformedPt.y);
				transformPositions[i++] = (float) tranformedPt.getX();
				transformPositions[i++] = (float) tranformedPt.getY();
				//transformPositions[i++] = (float) origPt.getX();
				//transformPositions[i++] = (float) origPt.getY();

			}

		}


		WarpGrid gw = new WarpGrid(0, 1, srcWidth-1, 0,1, srcHeight-1, transformPositions); // use warp grid becuse this uses hardware acceleration

		ParameterBlock pb = new ParameterBlock();
		pb.addSource(new BufferedImage(srcWidth,srcHeight,BufferedImage.TYPE_USHORT_GRAY));
		pb.add(gw);

//		pb.add(Interpolation.getInstance(Interpolation.INTERP_BICUBIC));
		warpOp =  JAI.create("warp", pb);

		/*

		System.out.println("Crop to:" + xOffset + ","+ yOffset + "  " + cropWidth + "x" + cropHeight );
		ParameterBlock cropPB = new ParameterBlock();
		cropPB.addSource(new BufferedImage(srcWidth,srcHeight,BufferedImage.TYPE_USHORT_GRAY));
		cropPB.add((float)xOffset);
		cropPB.add((float)yOffset);
		cropPB.add((float)cropWidth);
		cropPB.add((float)cropHeight);
		cropOp = JAI.create("crop",cropPB);
		 */
		/*
		ParameterBlock scalePB = new ParameterBlock();
		float  xScale = (float) outputWidth / (float) cropWidth;
		float  yScale = (float) outputHeight / (float) cropHeight;
		scalePB.add(xScale);
		scalePB.add((float) outputHeight / (float) cropHeight);
		scalePB.add(-xScale * (float) xOffset); // translate cropped section to 0,0
		scalePB.add(-yScale * (float) yOffset); 
		scalePB.addSource(cropOp);
		scaleOp = JAI.create("scale",scalePB);
		 */

		ParameterBlock scalePB = new ParameterBlock();
		float  xScale = (float) outputWidth / (float) srcWidth;
		float  yScale = (float) outputHeight / (float) srcHeight;
		scalePB.add(xScale);
		scalePB.add(yScale);
		scalePB.add(0f); // translate cropped section to 0,0
		scalePB.add(0f); 
		scalePB.addSource(warpOp);
		scaleOp = JAI.create("scale",scalePB);

		resetBackground(2);
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




	@Override
	public BufferedImage process(BufferedImage img) {
		if(mode == MODE.setCrop) {
			Graphics2D g = img.createGraphics();
			g.setColor(Color.GRAY);

			for(int i = 0; i < origTransformPoints.length -1; i++) {
				for(int j = 0; j < origTransformPoints[0].length-1 ; j++) {
					g.drawLine((int)mappedTransformPoints[i][j].x, (int)mappedTransformPoints[i][j].y, 
							(int)mappedTransformPoints[i][j+1].x, (int)mappedTransformPoints[i][j+1].y);

					g.drawLine((int)mappedTransformPoints[i][j].x,(int) mappedTransformPoints[i][j].y, 
							(int)mappedTransformPoints[i+1][j].x, (int)mappedTransformPoints[i+1][j].y);

				}
				g.drawLine((int)mappedTransformPoints[i][mappedTransformPoints[i].length-1].x,(int) mappedTransformPoints[i][mappedTransformPoints[i].length-1].y, 
						(int)mappedTransformPoints[i+1][mappedTransformPoints[i].length-1].x,(int) mappedTransformPoints[i+1][mappedTransformPoints[i].length-1].y);

			}

			Point.Float[] lastCol = mappedTransformPoints[mappedTransformPoints.length -1];
			for(int i = 0; i < lastCol.length -1; i++) {
				g.drawLine((int)lastCol[i].x,(int) lastCol[i].y,(int) lastCol[i+1].x,(int) lastCol[i+1].y);
			}			

			/*

			Enumeration<Point> e = mappedTransformPoints.elements();
			Point lastTop = e.nextElement();
			Point lastBottom = e.nextElement();

			g.drawLine(lastTop.x, lastTop.y, lastBottom.x, lastBottom.y);

			while(e.hasMoreElements()) {
				Point newTop = e.nextElement();
				Point newBottom = e.nextElement();
				g.drawLine(lastTop.x, lastTop.y, newTop.x, newTop.y);
				g.drawLine(lastBottom.x, lastBottom.y, newBottom.x, newBottom.y);
				g.drawLine(newTop.x, newTop.y, newBottom.x, newBottom.y);
				lastTop = newTop;
				lastBottom = newBottom;
			}
			 */

//			g.drawLine(topLeft.x, topLeft.y, topRight.x, topRight.y);
//			g.drawLine(topRight.x, topRight.y, bottomRight.x, bottomRight.y);
//			g.drawLine(bottomRight.x, bottomRight.y, bottomLeft.x, bottomLeft.y);
//			g.drawLine(bottomLeft.x, bottomLeft.y, topLeft.x, topLeft.y);
//			img.createGraphics().drawRect(xOffset, yOffset, cropWidth, cropHeight);			

			return img;			
		}

		warpOp.setSource(img, 0);

		//cropOp.setSource(img, 0); // automatically propagates to scaleOp
//		scaledImage.createGraphics().drawRenderedImage(scaleOp, new AffineTransform());

		scaledImage.createGraphics().drawRenderedImage(scaleOp,new AffineTransform());

		if(convertFromColor) {
			imageConversion.convertFromRGB(scaledImage, grayImage);			
		} else {
			imageConversion.convertFromGray(scaledImage, grayImage);
		}

		scaledImage.createGraphics().drawRenderedImage(grayImage, new AffineTransform());

		BufferedImage bkImage = background.update(grayImage);
		if(bkImage == null) return null;


		ImageDifference.apply(bkImage, grayImage, diffImage);

		thresh.apply(diffImage, threshImage);
		thresh2.apply(threshImage, threshImage2);

		if(theListener != null) {
			theListener.dataUpdate(threshImage2.getData());
		}


		switch(mode) {
		case raw:
			return img;
		case scaled:
			return scaledImage;
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

	public void setGridListener(GridListener gl) {
		theListener = gl;
	}
	/*
	public void setXOffset(int offset) {
		int newWidth = cropWidth + xOffset - offset;
		xOffset = offset;
		cropWidth =newWidth;
		resetCropAndScale();
	}


	public void setYOffset(int offset) {
		int newHeight = cropHeight + yOffset - offset;
		yOffset = offset;
		cropHeight = newHeight;
		resetCropAndScale();
	}


	public void setCropWidth(int cropWidth) {
		this.cropWidth = cropWidth;
		resetCropAndScale();
	}


	public void setCropHeight(int cropHeight) {
		this.cropHeight = cropHeight;
		resetCropAndScale();
	}

	public int getXOffset() {
		return xOffset;
	}

	public int getYOffset() {
		return yOffset;
	}

	public int getCropWidth() {
		return cropWidth;
	}

	public int getCropHeight() {
		return cropHeight;
	}
	 */


//	protected ImageMIPMap getMipMap(RenderedImage img) { 
	//	return new ImageMIPMap(img,  downScaleTransform,  Interpolation.getInstance(Interpolation.INTERP_BILINEAR)); 
//	}


}
