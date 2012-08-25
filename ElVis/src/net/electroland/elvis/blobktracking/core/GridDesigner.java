package net.electroland.elvis.blobktracking.core;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCircle;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_core.cvLine;
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_GRAY2RGB;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.ImageReceiver;
import net.electroland.elvis.imaging.imageFilters.GridUnwarp;
import net.electroland.elvis.util.CameraFactory;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class GridDesigner implements ImageReceiver, MouseMotionListener, MouseListener {
	
	GridUnwarp unwarp;
	
	CvPoint selectedPoint = null;
	CvPoint highlightedPoint = null;
	public static final int SEL_R = 7;
	public static final int SEL_R_SQR = SEL_R*SEL_R;
	IplImage img;
	IplImage lineImage;
	
	int imgWidth;
	int imgHeight;
	
	int gridWidth;
	int gridHeight;
	
	LinkedBlockingQueue<CvPoint[][]> gridQueue;
	
	CvPoint[][] curGrid;
	CvPoint[][] origGrid;
	CvScalar scalar;

	CanvasFrame canvas;
	ImageAcquirer camera;
	
	boolean isRunning = true;
	
	public GridDesigner(ElProps props) throws IOException, com.googlecode.javacv.FrameGrabber.Exception {
		canvas = new CanvasFrame("GridDesigner", 1);
		
		canvas.getCanvas().addMouseListener(this);
		canvas.getCanvas().addMouseMotionListener(this);
		
		gridWidth = props.getProperty("distortionGridWidth",  6);
		gridHeight = props.getProperty("distortionGridHeight", 4);
		
		imgWidth = props.getProperty("srcWidth", 640);
		imgHeight = props.getProperty("srcHeight", 480);

		gridQueue = new LinkedBlockingQueue<CvPoint[][]>();
		curGrid = makeGrid(gridWidth,gridHeight,imgWidth, imgHeight);
		origGrid = copyGrid(curGrid);
		unwarp = new GridUnwarp(imgWidth, imgHeight, curGrid, props);

		camera = CameraFactory.camera(props.getProperty("camera", CameraFactory.OPENCV_SRC),
				imgWidth,
				imgHeight,
				this);
		camera.start();
		
		new UpdateThread().start();
	}
	
	public class UpdateThread extends Thread {
		Vector<CvPoint[][]> grids = new Vector<CvPoint[][]>();
		public void run() {
			while(isRunning) {
				try {
					CvPoint[][]  lastGrid = gridQueue.take();
					gridQueue.drainTo(grids);
					if(! grids.isEmpty()) {
						lastGrid = grids.lastElement();
						grids.clear();
					}
					unwarp.updateGrid(lastGrid);
					
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static CvPoint[][] makeGrid(int gridWidth, int gridHeight, int imgWidth, int imgHeight) {
		System.out.println("makeGrid " + gridWidth + "x"+ gridHeight + "  "  + imgWidth + "x"+imgHeight);
		CvPoint[][] grid = new CvPoint[gridWidth+1][gridHeight+1];		
	
		float xStepSize = (float) imgWidth / (float) gridWidth;
		float yStepSize = (float) imgHeight / (float) gridHeight;
		
		for(int i = 0; i < gridWidth; i++) {
			for(int j=0; j < gridHeight; j++) {
				int x  = (int)(i * xStepSize);
				int y  = (int)(j * yStepSize);
				grid[i][j] = new CvPoint(x,y);
			}			
		}
		for(int i=0; i < gridWidth; i++) {
			grid[i][gridHeight] = new CvPoint((int) (i * xStepSize), imgHeight-1);
		}
		for(int j=0; j < gridHeight; j++) {
			grid[gridWidth][j] = new CvPoint(imgWidth-1, (int) (j * yStepSize));
		}
		grid[gridWidth][gridHeight] = new CvPoint(imgWidth-1, imgHeight-1);

		return grid;
		
	}
	
	public CvPoint[][] copyGrid(CvPoint[][] src) {
		CvPoint[][] dst = new CvPoint[src.length][src[0].length];
		for(int i = 0; i < src.length; i++) {
			for(int j = 0; j<src[0].length; j++) {
				dst[i][j] = new CvPoint(src[i][j]);
			}
		}
		return dst;
	}

	public static void main(String arg[]) throws IOException, com.googlecode.javacv.FrameGrabber.Exception {
		ElProps props;
		if(arg.length > 0) {
			props = ElProps.init(arg[0]);
		} else {
			props =ElProps.init("blobTracker.props");
		}
		
				
		


		new GridDesigner(
				props
		);
	}

	
	public void renderCanvas() {
		lineImage = (lineImage == null) ? cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 3 ) : lineImage;
		cvCvtColor(img,lineImage,CV_GRAY2RGB);
		CvScalar color = cvScalar(255,0,0,0);
		
		for(int i = 0; i <= gridWidth; i++) {
			CvPoint p1 = curGrid[i][0];
			for(int j = 0; j <= gridHeight; j++) {
				CvPoint p2 = curGrid[i][j];
//				System.out.println(p1 + " - " + p2);
				cvLine(lineImage, p1, p2, color, 1, 8, 0);
				p1 = p2;

			}
		}
		
		for(int j = 0; j <= gridHeight; j++) {
			CvPoint p1 = curGrid[0][j];
			for(int i = 0; i <= gridWidth; i++) {
				CvPoint p2 = curGrid[i][j];
				cvLine(lineImage, p1, p2, color, 1, 8, 0);
				p1 = p2;

			}
		}
		
		color = cvScalar(0,0,255,0);
		if(selectedPoint != null) {
			cvCircle(lineImage, selectedPoint, SEL_R, color, 1, 8, 0);
		}

		color = cvScalar(0,255,0,0);
		if(highlightedPoint != null) {
			cvCircle(lineImage, highlightedPoint, SEL_R, color, 1, 8, 0);
		}
		canvas.showImage(lineImage);
		if(canvas.isResizable()) canvas.setResizable(false);

		
		
	}
	public void addImage(IplImage i) {
		IplImage upwarped = unwarp.apply(i);
		img = upwarped.clone();
		renderCanvas();
	}

	@Override
	public void addImage(BufferedImage i) {
		addImage(IplImage.createFrom(i));
		
	}

	@Override
	public void receiveErrorMsg(Exception cameraException) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void mouseDragged(MouseEvent arg0) {
		if(selectedPoint != null) {
			selectedPoint.x(arg0.getX());
			selectedPoint.y(arg0.getY());
			unwarp.updateGrid(copyGrid(curGrid));
		}
		renderCanvas();
		
	}
	@Override
	public void mouseMoved(MouseEvent arg) {
		 highlightedPoint= getPoint(arg.getX(), arg.getY());
		 selectedPoint= null;
		
	}
	public void mouseClicked(MouseEvent arg0) {
	}
	
	@Override
	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	
	public void mousePressed(MouseEvent arg) {
		selectedPoint = getPoint(arg.getX(), arg.getY());
		highlightedPoint = null;
	}
	
	public CvPoint getPoint(int x, int y) {
		CvPoint closest= null;
		int minDist = Integer.MAX_VALUE;
		for(int i = 0; i <= gridWidth; i++) {
			for(int j=0; j <= gridHeight; j++) {
				CvPoint pt = curGrid[i][j];
				int dist = distSqr(x,y, pt);
				if(dist < minDist) {
					minDist = dist;
					closest = pt;
				}
			}			
		}
		if(minDist < SEL_R_SQR) {
			return closest;
		} else {
			return null;
		}
	}
	public int distSqr(int x, int y, CvPoint pt) {
		int dx = x - pt.x();
		int dy = y - pt.y();
		return dx*dx+dy*dy;
	}
	@Override
	public void mouseReleased(MouseEvent arg0) {
		selectedPoint = null;		
	}

	
}
