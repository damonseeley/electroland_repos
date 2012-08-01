package net.electroland.presenceGrid.ui;

import java.awt.Insets;
import java.awt.Point;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowEvent;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JMenuBar;

import net.electroland.presenceGrid.core.GridDetector.MODE;
import net.electroland.presenceGrid.util.GridProps;


public class GridFrame extends JFrame implements KeyListener, MouseListener, MouseMotionListener{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	GridPanel gridPanel;

	int mouseXOffset;
	int mouseYOffset;

	public GridFrame(String windowName) {	
		super(windowName);
		gridPanel = new GridPanel();		
		add(gridPanel);
		setSize(gridPanel.getSize().width + 40, gridPanel.getSize().height + 40);

		gridPanel.gridDetector.setThresh(GridProps.getGridProps().getProperty("threshold", 5000.0));
		gridPanel.gridDetector.setBackgroundAdaptation(GridProps.getGridProps().setProperty("adaptation", .001));



		setVisible(true);
		setResizable(true);

		addKeyListener(this);
		addMouseListener(this);
		addMouseMotionListener(this);


		addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(WindowEvent winEvt) {
				close();
			}
		});


		JMenuBar menubar = getJMenuBar();
		int mbh = (menubar != null ? menubar.getSize().height : 0);		
		Insets insets = getInsets();
		mouseXOffset = insets.left;
		mouseYOffset = insets.top +mbh;
	}

	public void close() {
		gridPanel.stop();

		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		} // wait a few second in case threads need to shut down nicely
		System.exit(0);
	}


	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_M:
			gridPanel.gridDetector.setThresh(gridPanel.gridDetector.getThresh() + 100);
			System.out.println("theshold increased to " + gridPanel.gridDetector.getThresh());
			GridProps.getGridProps().setProperty("threshold", gridPanel.gridDetector.getThresh());			
			break;
		case KeyEvent.VK_N:
			gridPanel.gridDetector.setThresh(gridPanel.gridDetector.getThresh() - 100);
			System.out.println("theshold decreased to " + gridPanel.gridDetector.getThresh());
			GridProps.getGridProps().setProperty("threshold", gridPanel.gridDetector.getThresh());
			break;
		case KeyEvent.VK_X:
			gridPanel.gridDetector.setBackgroundAdaptation(gridPanel.gridDetector.getAdaptation() + .0001);
			System.out.println("background adaptation increased to " + gridPanel.gridDetector.getAdaptation());
			GridProps.getGridProps().setProperty("adaptation", gridPanel.gridDetector.getAdaptation());			
			break;
		case KeyEvent.VK_Z:
			gridPanel.gridDetector.setBackgroundAdaptation(gridPanel.gridDetector.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " + gridPanel.gridDetector.getAdaptation());
			GridProps.getGridProps().setProperty("adaptation", gridPanel.gridDetector.getAdaptation());			
			break;
		}
	}


	public void keyReleased(KeyEvent e) {
		if(e.getKeyChar() == '?') {
			System.out.println("? - this help menu");
			System.out.println("s - save current properties to gridProps.props (will overwrite existing file)");
			System.out.println(". - view next window mode");
			System.out.println(", - view prev window mode");
			System.out.println("m - increase threshold");
			System.out.println("n - decrease threshold");
			System.out.println("x - increase background adaption rate");
			System.out.println("z - decrease background adaption rate");
			System.out.println("r - reset background model");
		}
		switch(e.getKeyCode()) {
		case KeyEvent.VK_S:
				GridProps.getGridProps().store();
			break;
		case KeyEvent.VK_PERIOD:
			gridPanel.gridDetector.nextMode();
			gridPanel.resetScaler();
			break;
		case KeyEvent.VK_COMMA:
			gridPanel.gridDetector.prevMode();
			gridPanel.resetScaler();
			break;			
		case KeyEvent.VK_R:
			gridPanel.gridDetector.resetBackground(3);
			break;



		}
	}

	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub

	}


	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub

	}


	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}


	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	Point.Float selectedPoint = null;

	public void updateCrop(MouseEvent e) {
		int x = (int) ((double) (e.getX()-mouseXOffset) * (1.0 / gridPanel.getScaler().getScaleX()));
		int y = (int) ((double) (e.getY()-mouseYOffset) * (1.0 / gridPanel.getScaler().getScaleY()));


		if(selectedPoint == null) {

			if(gridPanel.gridDetector.getMode() == MODE.setCrop) {
				selectedPoint = gridPanel.gridDetector.getSelectedPoint(x, y, 50);
			}
		}
		if(selectedPoint!= null) { // if point selected
			selectedPoint.setLocation(x, y);

		}

	}

	public void mousePressed(MouseEvent e) {
		updateCrop(e);
	}


	public void mouseReleased(MouseEvent e) {
		selectedPoint = null;
		gridPanel.gridDetector.resetCropAndScale();
		StringBuffer sb = new StringBuffer();
		for(int i = 0; i < gridPanel.gridDetector.mappedTransformPoints.length ; i++) {
			for(int j = 0; j < gridPanel.gridDetector.mappedTransformPoints[0].length; j++) {
				Point.Float pt = gridPanel.gridDetector.mappedTransformPoints[i][j];
				sb.append(pt.x);
				sb.append(",");
				sb.append(pt.y);
				sb.append(",");
			}

		}
		sb.deleteCharAt(sb.length()-1);
		gridPanel.props.setProperty("transformMapping", sb.toString());

	}

	public void mouseDragged(MouseEvent e) {
		updateCrop(e);		
	}

	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public  GridPanel getGridPanel() {
		return gridPanel;
	}

}
