package net.electroland.presenceGrid.ui;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JPanel;
import javax.swing.Timer;

import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.presenceGrid.core.GridDetector;
import net.electroland.presenceGrid.core.GridDetector.MODE;
import net.electroland.presenceGrid.util.GridProps;


public class GridPanel extends JPanel implements ActionListener {

	public static final String JMYRON_SRC = "jMyronCam";
	public static final String LOCALAXIS_SRC ="axis";


	int srcImageWidth;
	int srcImageHeight;
	int displayWidth;
	int displayHeight;

	int gridWidth;
	int gridHeight;


	AffineTransform imageScaler;

	ImageAcquirer srcStream;
	BufferedImage srcImage;

	GridProps props = GridProps.getGridProps();

	public GridDetector gridDetector;

	public GridPanel() {

		srcImageWidth = props.getProperty("srcImageWidth", 240);
		srcImageHeight = props.getProperty("srcImageHeight", 180);

		displayWidth = props.getProperty("displayWidth", 480);
		displayHeight = props.getProperty("displayHeight", 360);

		gridWidth = props.getProperty("gridWidth", 65);
		gridHeight = props.getProperty("gridHeight", 45);






		gridDetector = new GridDetector(srcImageWidth, srcImageHeight, gridWidth, gridHeight);

		Point transformCells = props.getProperty("transformDims", new Point(3,2));
		String cellMapStr = props.getProperty("transformMapping", "");
		
		String[] cellMapStrArg  = cellMapStr.split(",");
	
		if(cellMapStrArg.length != 2 * (transformCells.x + 1) * (transformCells.y + 1)) {
			System.out.println("Reconstructing transfrom grid from scratch only " + cellMapStrArg.length + " element in prop");
					gridDetector.resetWarpGrid(transformCells.x, transformCells.y);
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
			gridDetector.resetWarpGrid(map);

		}


		resetScaler();

		gridDetector.start();


		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();

		setSize(displayWidth, displayHeight);
		setPreferredSize(new Dimension(displayWidth, displayHeight));

		try {
			setSourceStream(props.getProperty("camera", LOCALAXIS_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public void resetScaler() {
		if((gridDetector.getMode() == MODE.setCrop) ||  (gridDetector.getMode() == MODE.raw)){
			imageScaler =  new AffineTransform();
			imageScaler.scale( ((float) displayWidth) / ((float) srcImageWidth),((float) displayHeight) / ((float) srcImageHeight) );
		} else {
			imageScaler =  new AffineTransform();
			imageScaler.scale( ((float) displayWidth) / ((float) gridWidth),((float) displayHeight) / ((float) gridHeight) );
		}

	}

	public void setSourceStream(String s) throws IOException {
		if(srcStream != null) {
			srcStream.stopRunning();
		}

//		int frameSkip = 2;
		if(s.equals(JMYRON_SRC)) {
//			frameSkip = 50;
			srcStream = new WebCam(srcImageWidth, srcImageHeight, 12, gridDetector, false);
		} else if(s.equals(LOCALAXIS_SRC)) {
			String ip = props.getProperty("axisIP", "10.0.1.90");		
			String url = "http://" + ip + "/";
			String username = props.getProperty("axisUsername", "root");
			String password = props.getProperty("axisPassword", "n0h0");
			srcStream = new AxisCamera(url, srcImageWidth, srcImageHeight, 0, 0 , username, password, gridDetector);
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

	public AffineTransform getScaler() {
		return imageScaler;
	}
	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2d = (Graphics2D)g;

		RenderedImage ri=null;
		if(srcStream != null) {
			ri = gridDetector.getImage();
		} else if (srcImage != null) {
			ri = srcImage;
		}
		if (ri != null) {
			g2d.drawRenderedImage(ri,  imageScaler);
		}

		renderDrawing(g2d);
	}

	public void renderDrawing(Graphics2D g2d) {
	}


	public void actionPerformed(ActionEvent e) {
		repaint();				
	}

	public void stop() {
		srcStream.stopRunning();
	}


}
