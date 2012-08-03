package net.electroland.elvis.manager;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.AffineTransform;
import java.io.File;
import java.io.IOException;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.Timer;

import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.FlyCapture.FlyCamera;
import net.electroland.elvis.imaging.acquisition.axisCamera.FlowerCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.LocalCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NavyCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoSouthCam;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvis.regions.PolyRegion;

import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
public class ImagePanel extends JPanel implements MouseListener, MouseMotionListener, KeyListener, Colorable, ActionListener {

	public static ImagePanel THE_IMAGEPANEL;
//	CanvasFrame canvasFrame;
	boolean aquireInColor = false;
	public static final String NAVY_SRC = "Navy St.";
	public static final String FLOWER_SRC = "Flower St.";
	public static final String NOHOSOUTH_SRC = "NoHo South";
	public static final String NOHONORTH_SRC = "NoHo North";
	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="Local Axis";
	public static final String FLY_SRC = "Fly Cam";

	public static final String RAW_IMG = "Raw";
	public static final String GRAYSCALE_IMG = "Grayscale";
	public static final String BLUR_IMG = "Blur";
	public static final String BACKGROUND_IMG = "Background";
	public static final String BACKDIFF_IMG = "Difference";
	public static final String THRESHOLD_IMG = "Threshold";
	public static final String CONTOUR_IMG = "Contour";


//	public static ImagePanel THE_IMAGEPANEL = null;
	PolyRegion selectedRegion = null;

	boolean shiftDown = false;
	boolean ctrlDown = false;

	//ds set to 2 for easier tiny region editing
	public static final int DIST_RADIUS_HALF = 2;
	public static final int DIST_RADIUS = DIST_RADIUS_HALF + DIST_RADIUS_HALF;
	public static final int DIST_RADIUS_SQR = DIST_RADIUS * DIST_RADIUS; // pixels square

	int xOffset;
	int yOffset;

	int mouseX;
	int mouseY;

	int lastX;
	int lastY;

	boolean mouseInFrame;

	ImageAcquirer srcStream;
	IplImage srcImage;

	int w ;
	int h ;
	
	public static int SCALE = 1;
	public static AffineTransform SCALER ;	
	public static double INV_SCALER = 1.0 / (double) SCALE	;

	PresenceDetector presenceDetector;

	PolyRegion curPolyRegion = null;


	public Vector<PolyRegion> regions = new Vector<PolyRegion>();
	Color curColor = Color.RED;

	PolyRegion.DistResult draggedPoint = null;
	PolyRegion draggedRegion = null;

	JButton editToggleButton; 
	boolean editMode = true;

	
	public ImagePanel() {
		this(160,120);
	}
	
	public ImagePanel(int w, int h) {
		this.w = w;
		this.h = h;
		// adjust scale?
		ImagePanel.THE_IMAGEPANEL = this;
//		canvasFrame = new CanvasFrame("Elvis");
//		THE_IMAGEPANEL = this;
		presenceDetector = new PresenceDetector(w,h);
		presenceDetector.start();
		presenceDetector.setRegions(null);
		setSize(w*SCALE,h*SCALE);
		SCALER =  new AffineTransform();
		SCALER.scale(SCALE,SCALE);
		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();

		setFocusable(true); 
		addKeyListener(this);

		setLayout(new BorderLayout());

		setPreferredSize(new Dimension(w * SCALE, h * SCALE));


		editToggleButton = new JButton("Test Regions");
		editToggleButton.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				//TODO:
				ImagePanel.THE_IMAGEPANEL.toggleEditMode();
			}
		});
		JPanel bottomPanel = new JPanel();
		
		bottomPanel.setLayout(new BorderLayout());

		JLabel lable = new JLabel(
				"<html><ol>" +
				"<li>Select Source</li>"+
				"<li>Click in image to add points</li>"+
				"<li>Close polygon to create region</li>"+
				"<li>Click and drag to move vertex or region</li>"+
				"<li>Shift-Click on vertex to delete vertex</li>"+
				"<li>Shift-Click on edge to add vertex</li>"+
				"<li>Shift-Click on region to delete</li>"+
				"<li>Ctrl-Click on region to select</li>"+
				"<li>Modify region properties</li>"+
				"<li>Click \"Test Regions\" to test</li>"+
				"<li>region settings can be modified while testing(not geometry)</li>"+
				"<li>Click \"Edit Regions\" to resume editing</li>"+				
				"</ol></html>");

		bottomPanel.add(lable, BorderLayout.NORTH);

		bottomPanel.add(editToggleButton, BorderLayout.SOUTH);
		
		
		
		add(bottomPanel, BorderLayout.SOUTH);

	}

	public void toggleEditMode() {
		editMode = ! editMode;
		if(editMode) {
			editToggleButton.setText("Test Regions");
			presenceDetector.setRegions(null);
		} else {
			editToggleButton.setText("Edit Regions");
			presenceDetector.setRegions(regions);
		}

	}
	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2d = (Graphics2D)g;

		IplImage ri=null;
		if(srcStream != null) {
			ri = presenceDetector.getImageIplImage();
		} else if (srcImage != null) {
			ri = srcImage;
		}
		if (ri != null) {
			g2d.drawRenderedImage(ri.getBufferedImage(), SCALER);
//			canvasFrame.showImage(ri);
		}

		renderDrawing(g2d);
	}

	public void  moveUp(PolyRegion pr) {		
		moveUp(regions.indexOf(pr));
	}

	public void  moveDown(PolyRegion pr) {
		moveDown(regions.indexOf(pr));
	}
	public int getDepth(PolyRegion pr) {
		return regions.indexOf(pr);
	}
	public void  moveDown(int i) {
		if (i < 1) return;
		if (i > regions.size() - 1) return;
		PolyRegion mDown = regions.get(i);
		PolyRegion mUp= regions.get(i-1);

		regions.set(i, mUp);
		regions.set(i-1, mDown);
	}

	public void move(PolyRegion r, int i) {
		regions.remove(r);
		regions.insertElementAt(r, i);
	}

	public void  moveUp(int i) {
		if (i < 0) return;
		if (i >= regions.size() - 1) return;

		PolyRegion mUp = regions.get(i);
		PolyRegion mDown = regions.get(i+1);
		regions.set(i, mDown);
		regions.set(i+1, mUp);

	}

	public void setThresh(double d) {
		presenceDetector.setThresh(d);
	}

	public double getThresh() {
		return presenceDetector.getThresh();
	}
	
	public void stop() {
		presenceDetector.stopRunning();
		try {
			Thread.sleep(100); // wait for image to be added to queue so will exit
		} catch (InterruptedException e) {
		} 
		if(srcStream != null) {
			srcStream.stopRunning();
		}
		
	}
	public void setBackgroundStream(String s) throws IOException {
		if(srcStream != null) {
			srcStream.stopRunning();
		}
		int frameSkip = 2;
		if(s.equals(NAVY_SRC)) {
			srcStream = new NavyCam(w,h,presenceDetector, aquireInColor);
		} else if(s.equals(FLOWER_SRC)) {
			srcStream = new FlowerCam(w,h,presenceDetector, aquireInColor);
		} else if(s.equals(NOHOSOUTH_SRC)) {
				srcStream = new NoHoSouthCam(w,h,presenceDetector, aquireInColor);
		} else if(s.equals(NOHONORTH_SRC)) {
			srcStream = new NoHoNorthCam(w,h,presenceDetector, aquireInColor);
		} else if(s.equals(JMYRON_SRC)) {
			frameSkip = 50;
			srcStream = new WebCam(w, h, 12, presenceDetector, aquireInColor);
		} else if(s.equals(LOCALAXIS_SRC)) {
			System.out.println("creating local " + w +"x" +h);
			srcStream = new LocalCam(w,h,presenceDetector, aquireInColor);
		} else if(s.equals(FLY_SRC)) {
			System.out.println("creating fly camera " + w +"x" +h);
			try {
				srcStream = new FlyCamera(presenceDetector, 0 , w, h);
			} catch (Exception e) {
				srcStream = null;
				e.printStackTrace();
			}
		}else {
			srcStream = null;
			throw new IOException("Unknown source");
		}
		srcStream.start();
		presenceDetector.resetBackground(frameSkip);
	}
	public void setBackgroundImage(File f) throws IOException {
		if(srcStream != null) {
			srcStream.stopRunning();
			srcStream = null;
		}
		//TOOD: shoudl lod this with cvLoadImage (which I can't find at the moment

		srcImage = 	IplImage.createFrom(ImageIO.read(f));
	}

	public void renderDrawing(Graphics2D g2d) {
		if(		selectedRegion != null) 
			selectedRegion.isSelected= true;

		for(PolyRegion pr : regions) {
			pr.render(g2d);
		}

		if(curPolyRegion != null) {
			curPolyRegion.render(g2d);
			if (curPolyRegion.startPointInRange(mouseX, mouseY, DIST_RADIUS_SQR)) {
				curPolyRegion.drawHighlightedPoint(g2d, 0, DIST_RADIUS_HALF, DIST_RADIUS);
			}
		} else {
			PolyRegion.DistResult prd = getNearestPoint(mouseX, mouseY);
			if(prd != null) {
				prd.region.drawHighlightedPoint(g2d, prd.index, DIST_RADIUS_HALF, DIST_RADIUS);
			} else {
				if(shiftDown) {
					prd = getNearestEdge(mouseX, mouseY);
					if(prd != null) {
						PolyRegion.drawHighlightedPoint(g2d, mouseX, mouseY, DIST_RADIUS_HALF, DIST_RADIUS, Color.GRAY);
					}
				}
			}
		}
	}

	public void calcMouseLoc(MouseEvent e) {
		mouseX = e.getX() - getX() -xOffset;
		mouseY = e.getY() - getY() - yOffset;

		if(mouseX < 0) {
			mouseInFrame = false;
			return;
		} else if(mouseX >= getWidth()) {
			mouseInFrame = false;			
			return;
		}
		if(mouseY < 0) {
			mouseInFrame = false;
			return;			
		} else if (mouseY >= getHeight()) {
			mouseInFrame = false;
			return;
		}
		mouseInFrame = true;
	}

	public void mouseDragged(MouseEvent e) {
		if(! editMode) return;
		lastX = mouseX;
		lastY = mouseY;
		calcMouseLoc(e);
		if(! mouseInFrame) return;
		if(curPolyRegion == null) {
			if((draggedPoint == null) && (draggedRegion == null)) {
				draggedPoint = getNearestPoint(mouseX, mouseY);
				if(draggedPoint == null) {
					draggedRegion = getClickedRegion(mouseX, mouseY);
					lastX = mouseX;
					lastY = mouseY;
				}
			}
			if(draggedPoint != null) {
				draggedPoint.region.movePoint(draggedPoint.index, mouseX, mouseY);
			} else if(draggedRegion != null) {
				draggedRegion.translate(mouseX-lastX, mouseY-lastY);
				lastX = mouseX;
				lastY = mouseY;
			}
		}
		repaint();
	}


	public void mouseMoved(MouseEvent e) {
		if(! editMode) return;

		calcMouseLoc(e);

		if(! mouseInFrame) return;
		if(curPolyRegion != null) {
			curPolyRegion.moveLastPoint(mouseX, mouseY);
		}
		repaint();

	}

	public void selectRegion(PolyRegion r) {
		selectedRegion = r;
		RegionSettingsPanelMig.thePanel.updateForNewDisplay();
	}

	public void mouseClicked(MouseEvent e) {

		calcMouseLoc(e);
		if(! mouseInFrame) return;
		if(ctrlDown) {
			curPolyRegion = null;
			PolyRegion r  = getClickedRegion(mouseX, mouseY);
			if(r != null) {
				selectRegion(r);
			}

		} else {
			if(! editMode) return;
			if(shiftDown) { // if shift is down delete of add point if approtiate

				curPolyRegion = null;
				PolyRegion.DistResult result = getNearestPoint(mouseX, mouseY);
				if(result != null) {
					result.region.removePoint(result.index);
					if(result.region.size() <= 2) {
						System.out.println("removing");
						regions.remove(result.region);
					}

				} else {
					PolyRegion.DistResult  prd = getNearestEdge(mouseX, mouseY);
					if(prd != null) {
						prd.region.insertPoint(prd.index, mouseX, mouseY);
					} else {
						PolyRegion r  = getClickedRegion(mouseX, mouseY);
						if(r != null) {
							regions.remove(r);
						}

					}
				}
			} else if(curPolyRegion == null) {
				curPolyRegion = new PolyRegion();
				curPolyRegion.setColor(curColor);
				curPolyRegion.addPoint(mouseX, mouseY);
				curPolyRegion.addPoint(mouseX, mouseY); // last point gets moved
			} else {
				if(curPolyRegion.startPointInRange(mouseX, mouseY, DIST_RADIUS_SQR)) {
					curPolyRegion.removeLastPoint();
					curPolyRegion.isFilled = true; // done
					regions.add(curPolyRegion);
					selectRegion(curPolyRegion);
					curPolyRegion = null;
				} else {
					curPolyRegion.moveLastPoint(mouseX, mouseY);
					curPolyRegion.addPoint(mouseX, mouseY);
				}
			}
		}
		repaint();

	}
	public void mouseEntered(MouseEvent e) {
		requestFocus();


	}
	public void mouseExited(MouseEvent e) {

	}
	public void mousePressed(MouseEvent e) {

	}
	public void mouseReleased(MouseEvent e) {
		draggedPoint = null;
		draggedRegion = null;

	}

	public void setImageViewType(String s) {
		if(s.equals(RAW_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.RAW);			
		} else if (s.equals(GRAYSCALE_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.GRAY);						
		} else if(s.equals(BACKGROUND_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.BGRND);						
		} else if(s.equals(BACKDIFF_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.DIFF);			
		} else if(s.equals(THRESHOLD_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.THRESH);			
		} else if (s.equals(CONTOUR_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.CONTOUR);
		} else if (s.equals(BLUR_IMG)) {
			presenceDetector.setImageReturn(PresenceDetector.ImgReturnType.BLUR);			
		}

	}

	public PolyRegion.DistResult getNearestPoint(int x, int y) {
		PolyRegion.DistResult nearPoint = null;
		for(PolyRegion r : regions) {
			PolyRegion.DistResult tmp = r.getNearestPoint(x, y, DIST_RADIUS, DIST_RADIUS_SQR);
			if(tmp != null) {
				if((nearPoint == null) || (tmp.distSquared < nearPoint.distSquared)) {
					nearPoint = tmp;
				}
			}
		}
		return nearPoint;

	}
	public PolyRegion.DistResult getNearestEdge(int x, int y) {
		PolyRegion.DistResult nearEdge = null;
		for(PolyRegion r : regions) {
			PolyRegion.DistResult tmp = r.getNearestEdge(x, y, DIST_RADIUS, DIST_RADIUS_SQR);
			if(tmp != null) {
				if((nearEdge == null) || (tmp.distSquared < nearEdge.distSquared)) {
					nearEdge = tmp;
				}
			}
		}
		return nearEdge;

	}
	public PolyRegion getClickedRegion(int x, int y) {
		PolyRegion curRegion = null;
		for(PolyRegion r : regions) {
			if(r.contains(x, y)) {
				curRegion = r;
			}
		}
		return curRegion;

	}

	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_SHIFT:
			shiftDown = true;
			curPolyRegion  = null;
			selectRegion(null);
			break;
		case KeyEvent.VK_CONTROL:
		case KeyEvent.VK_ALT:
			ctrlDown = true;
			curPolyRegion  = null;
			break;
		}
		repaint();

	}

	public void keyReleased(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_SHIFT:
			shiftDown = false;			
			break;
		case KeyEvent.VK_CONTROL:
		case KeyEvent.VK_ALT:
			ctrlDown = false;
			break;
		}
		repaint();


	}

	public void setAdaptation(double d) {
		presenceDetector.setAdaptation(d);		
	}
	public double getAdaptation() {
		return presenceDetector.getAdaptation();		
	}

	public void keyTyped(KeyEvent e) {

	}

	public void setColor(Color c) {
		curColor = c;

	}
	public Color getColor() {
		return curColor;
	}

	public void actionPerformed(ActionEvent e) {
		repaint();		
	}



}
