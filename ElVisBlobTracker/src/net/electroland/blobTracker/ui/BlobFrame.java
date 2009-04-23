package net.electroland.blobTracker.ui;

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

import net.electroland.blobTracker.core.BlobTracker;
import net.electroland.blobTracker.util.ElProps;


public class BlobFrame extends JFrame implements  KeyListener, MouseListener, MouseMotionListener{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public BlobPanel blobPanel;

	int mouseXOffset;
	int mouseYOffset;
	

	public BlobFrame(String windowName, BlobTracker blobTracker) {	
		super(windowName);
		blobPanel = new BlobPanel(blobTracker);		
		add(blobPanel);
		setSize(blobPanel.getSize().width + 40, blobPanel.getSize().height + 40);

		
	
		blobPanel.trackListeners = new BlobPanel.SimpleTrackListener[ blobPanel.blobTracker.regionMap.size()];
		for(int i = 0; i < blobPanel.blobTracker.regionMap.size(); i++) {
			blobPanel.trackListeners[i] = new BlobPanel.SimpleTrackListener();
			blobPanel.blobTracker.tracker[i].addListener(blobPanel.trackListeners[i]);
		}



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
		System.out.println("** Type ? for help **");
	}

	public void close() {
		blobPanel.stop();

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

	public  BlobPanel getBlobPanel() {
		return blobPanel;
	}

	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_M:
			blobPanel.blobTracker.setThresh(blobPanel.blobTracker.getThresh() + 100);
			System.out.println("theshold increased to " + blobPanel.blobTracker.getThresh() );
			ElProps.THE_PROPS.setProperty("threshold", blobPanel.blobTracker.getThresh() );			
			break;
		case KeyEvent.VK_N:
			blobPanel.blobTracker.setThresh(blobPanel.blobTracker.getThresh() - 100);
			System.out.println("theshold decreased to " + blobPanel.blobTracker.getThresh());
			ElProps.THE_PROPS.setProperty("threshold", blobPanel.blobTracker.getThresh());
			break;
		case KeyEvent.VK_X:
			blobPanel.blobTracker.setBackgroundAdaptation(blobPanel.blobTracker.getAdaptation() + .0001);
			System.out.println("background adaptation increased to " + blobPanel.blobTracker.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", blobPanel.blobTracker.getAdaptation());			
			break;
		case KeyEvent.VK_Z:
			blobPanel.blobTracker.setBackgroundAdaptation(blobPanel.blobTracker.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " + blobPanel.blobTracker.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", blobPanel.blobTracker.getAdaptation());			
			break;
			
		// ds additions	
		case KeyEvent.VK_P:
			blobPanel.blobTracker.regionMap.incProvisionalPenalty0();
			System.out.println("provisionalPenalty0 increased to " + blobPanel.blobTracker.regionMap.getProvisionalPenalty0());
			break;
		case KeyEvent.VK_O:
			blobPanel.blobTracker.regionMap.decProvisionalPenalty0();
			System.out.println("provisionalPenalty0 decreased to " + blobPanel.blobTracker.regionMap.getProvisionalPenalty0());
			break;
		case KeyEvent.VK_T:
			blobPanel.blobTracker.regionMap.decMaxTrackMove0();
			System.out.println("maxTrackMove0 decreased to " + blobPanel.blobTracker.regionMap.getMaxTrackMove0());
			break;
		case KeyEvent.VK_Y:
			blobPanel.blobTracker.regionMap.incMaxTrackMove0();
			System.out.println("maxTrackMove0 increased to " + blobPanel.blobTracker.regionMap.getMaxTrackMove0());
			break;
		case KeyEvent.VK_U:
			blobPanel.blobTracker.regionMap.decNonMatchPenalty0();
			System.out.println("nonMatchPenalty0 decreased to " + blobPanel.blobTracker.regionMap.getNonMatchPenalty0());
			break;
		case KeyEvent.VK_I:
			blobPanel.blobTracker.regionMap.incNonMatchPenalty0();
			System.out.println("nonMatchPenalty0 increased to " + blobPanel.blobTracker.regionMap.getNonMatchPenalty0());
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
			System.out.println("m - increase threshold");
			System.out.println("n - decrease threshold");
			System.out.println("x - increase background adaption rate");
			System.out.println("z - decrease background adaption rate");
			System.out.println("r - reset background model");
			System.out.println("t - decrease maxTrackMove for region0");
			System.out.println("y - increase maxTrackMove for region0");
			System.out.println("u - decrease nonMatchPenalty for region0");
			System.out.println("i - increase nonMatchPenalty for region0");
			System.out.println("o - decrease provisionalPenalty for region0");
			System.out.println("p - increase provisionalPenalty for region0");
			
			break;
		case 's':
			ElProps.THE_PROPS.store();
			break;	
		case '.':
			blobPanel.blobTracker.nextMode();
			break;
		case ',':
			blobPanel.blobTracker.prevMode();
			break;
		case 'r':
			blobPanel.blobTracker.resetBackground(3);
			break;

			
		}
	}

	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}

}
