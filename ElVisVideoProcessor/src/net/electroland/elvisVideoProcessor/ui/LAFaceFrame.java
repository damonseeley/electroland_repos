package net.electroland.elvisVideoProcessor.ui;

import java.awt.Insets;
import java.awt.Point;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;
import javax.swing.JMenuBar;

import net.electroland.elvisVideoProcessor.ElProps;
import net.electroland.elvisVideoProcessor.LAFaceVideoProcessor;


public class LAFaceFrame extends JFrame implements  KeyListener, MouseListener, MouseMotionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public LAFacePanel facePanel;
	public LAFaceVideoProcessor vidProcessor;
	int mouseXOffset;
	int mouseYOffset;

//	CurveEditor curveEditor;



	public LAFaceFrame(String windowName, LAFaceVideoProcessor vidProcessor, ElProps props) {	
		super(windowName);
		facePanel = new LAFacePanel(vidProcessor);		
		this.vidProcessor = vidProcessor;
		add(facePanel);
		setSize(facePanel.getSize().width + 40, facePanel.getSize().height + 40);





		setVisible(true);
		setResizable(true);

		//	addKeyListener(this);
		addMouseListener(this);

//		GridDetectorServer.console.addKeyListener(this);

		addMouseMotionListener(this);
		addKeyListener(this);


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

		String curveFile = props.getProperty("curveFile", "");
//		CurveEditor.CEStarter starter = new CurveEditor.CEStarter(curveFile);
//		starter.start();
//		curveEditor = starter.getEditor();

		System.out.println("** Type ? for help **");

	}

	public void close() {
		facePanel.stop();

		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		} // wait a few second in case threads need to shut down nicely
		System.exit(0);
	}




	public void mouseClicked(MouseEvent e) {

	}


	public void mouseEntered(MouseEvent e) {

	}


	public void mouseExited(MouseEvent e) {

	}

	Point.Float selectedPoint = null;



	public void mousePressed(MouseEvent e) {
	}


	public void mouseReleased(MouseEvent e) {

	}

	public void mouseDragged(MouseEvent e) {
	}

	public void mouseMoved(MouseEvent e) {

	}

	public  LAFacePanel getFacePanel() {
		return facePanel;
	}

	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {

		case KeyEvent.VK_X:
			vidProcessor.setBackgroundAdaptation(vidProcessor.getAdaptation() + .0001);
			System.out.println("background adaptation increased to " + vidProcessor.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", vidProcessor.getAdaptation());			
			break;
		case KeyEvent.VK_Z:
			vidProcessor.setBackgroundAdaptation(vidProcessor.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " + vidProcessor.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", vidProcessor.getAdaptation());			
			break;
		}
	}

	public void keyReleased(KeyEvent e) {
		switch(e.getKeyChar()) {
		case '?':
			System.out.println("");
			System.out.println("? - this help menu");
			System.out.println("s - save current properties to gridProps.props (will overwrite existing file)");
			System.out.println(". - view next window mode");
			System.out.println(", - view prev window mode");
			System.out.println("x - increase background adaption rate");
			System.out.println("z - decrease background adaption rate");
			System.out.println("r - reset background model");

			break;
		case 's':
			ElProps.THE_PROPS.store();
			break;	
		case '.': {
			preModeChange();
			LAFaceVideoProcessor.MODE mode = vidProcessor.getMode();
			vidProcessor.nextMode();
			postModeChange(mode);
			repaint();
		};
		break;
		case ',': {
			preModeChange();
			LAFaceVideoProcessor.MODE mode = vidProcessor.getMode();
			vidProcessor.prevMode();
			postModeChange(mode);
			repaint();
		};
		break;
		case 'r':
			vidProcessor.resetBackground(3);
			break;


		}
	}

	public void preModeChange() {
		if(vidProcessor.getMode() == LAFaceVideoProcessor.MODE.setWarp) {
			facePanel.removeMouseListener(vidProcessor.getROIConstructor());
			facePanel.removeMouseMotionListener(vidProcessor.getROIConstructor());
		} else if 		(vidProcessor.getMode() == LAFaceVideoProcessor.MODE.mosaic) {
			facePanel.removeMouseListener(vidProcessor.getMosaicConstructor());
			facePanel.removeMouseMotionListener(vidProcessor.getMosaicConstructor());
		}
		
	}
	public void postModeChange(LAFaceVideoProcessor.MODE prevMode) {
		if(vidProcessor.getMode() == LAFaceVideoProcessor.MODE.setWarp) {
			facePanel.addMouseListener(vidProcessor.getROIConstructor());
			facePanel.addMouseMotionListener(vidProcessor.getROIConstructor());
		}	else if (prevMode == LAFaceVideoProcessor.MODE.setWarp) {
			vidProcessor.resetWarpAndROI();
			ElProps.THE_PROPS.setProperty("warpGrid", vidProcessor.getROIConstructor().toString());
		} 
		
		if(vidProcessor.getMode() == LAFaceVideoProcessor.MODE.mosaic) {
			facePanel.addMouseListener(vidProcessor.getMosaicConstructor());
			facePanel.addMouseMotionListener(vidProcessor.getMosaicConstructor());
		} else if (prevMode == LAFaceVideoProcessor.MODE.mosaic) {
			ElProps.THE_PROPS.setProperty("mosaicRects", vidProcessor.getMosaicConstructor().toString());			
		}
		
		
		/*
		
		if(vidProcessor.getMode() == LAFaceVideoProcessor.MODE.colorAdjust) {
			curveEditor.addLutChangeListener(this);
			curveEditor.setVisible(true);
		} else if (prevMode == LAFaceVideoProcessor.MODE.colorAdjust) {
			curveEditor.removeLutChangeListener(this);
			curveEditor.setVisible(false);
		}
		*/
		facePanel.setModeString(vidProcessor.getMode().toString());
		
	}

	public void keyTyped(KeyEvent e) {
	}

//	public void lutChanged() {
//		vidProcessor.setLutCache(curveEditor.getLut(65536));
//	}

}
