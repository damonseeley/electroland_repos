package net.electroland.elvis.blobktracking.ui;

import java.awt.Insets;
import java.awt.Point;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.JFrame;
import javax.swing.JMenuBar;

import net.electroland.elvis.blobtracking.BlobTracker;
import net.electroland.elvis.util.ElProps;


public class BlobFrame extends JFrame implements MouseListener, MouseMotionListener{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public BlobPanel blobPanel;

	int mouseXOffset;
	int mouseYOffset;
	
	ElProps props;
	

	public BlobFrame(ElProps props, String windowName, BlobTracker blobTracker) {	
		super(windowName);
		this.props = props;
		blobPanel = new BlobPanel(props, blobTracker);		
		add(blobPanel);
		setSize(blobPanel.getSize().width + 40, blobPanel.getSize().height + 40);

		
	
		blobPanel.trackListeners = new BlobPanel.SimpleTrackListener[ 1];
//		for(int i = 0; i < blobPanel.blobTracker.regionMap.size(); i++) {
			blobPanel.trackListeners[0] = new BlobPanel.SimpleTrackListener();
			blobPanel.blobTracker.tracker.addListener(blobPanel.trackListeners[0]);
//			PresenceDetector..addListener(blobPanel.trackListeners[0]);
	//	}



		setVisible(true);
		setResizable(true);

	//	addKeyListener(this);
		addMouseListener(this);

//		GridDetectorServer.console.addKeyListener(this);

		addMouseMotionListener(this);
		addKeyListener(new PresenceDetectorKeyListener(props, blobTracker.presenceDetector));




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
/*
	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {
		case KeyEvent.VK_M:
			blobPanel.blobTracker.presenceDetector.setThresh(blobPanel.blobTracker.presenceDetector.getThresh() + 1);
			System.out.println("theshold increased to " + blobPanel.blobTracker.presenceDetector.getThresh() );
			props.setProperty("threshold",blobPanel.blobTracker.presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_N:
			blobPanel.blobTracker.presenceDetector.setThresh(blobPanel.blobTracker.presenceDetector.getThresh() -1);
			System.out.println("theshold increased to " + blobPanel.blobTracker.presenceDetector.getThresh() );
			props.setProperty("threshold",blobPanel.blobTracker.presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_X:
			blobPanel.blobTracker.presenceDetector.setThresh(blobPanel.blobTracker.presenceDetector.getAdaptation() + .0001);
			System.out.println("adaptation increased to " + blobPanel.blobTracker.presenceDetector.getAdaptation() );
			props.setProperty("adaptation",blobPanel.blobTracker.presenceDetector.getAdaptation() );			
		break;
		case KeyEvent.VK_Z:
			blobPanel.blobTracker.presenceDetector.setThresh(blobPanel.blobTracker.presenceDetector.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " +blobPanel.blobTracker.presenceDetector.getAdaptation());
			props.setProperty("adaptation", blobPanel.blobTracker.presenceDetector.getAdaptation());			
			break;
			
		// ds additions	
		case KeyEvent.VK_P:
			blobPanel.blobTracker.tracker.csp.setProvisionalPenalty(blobPanel.blobTracker.tracker.csp.getProvisionalPenalty() + 1);
			System.out.println("provisionalPenalty  increased to " +blobPanel.blobTracker.tracker.csp.getProvisionalPenalty());
			props.setProperty("provisionalPenalty", blobPanel.blobTracker.tracker.csp.getProvisionalPenalty());			
			break;
		case KeyEvent.VK_O:
			blobPanel.blobTracker.tracker.csp.setProvisionalPenalty(blobPanel.blobTracker.tracker.csp.getProvisionalPenalty() - 1);
			System.out.println("provisionalPenalty  decreased to " +blobPanel.blobTracker.tracker.csp.getProvisionalPenalty());
			props.setProperty("provisionalPenalty", blobPanel.blobTracker.tracker.csp.getProvisionalPenalty());			
			break;
			// these are not easy to change while running
		case KeyEvent.VK_T:
			blobPanel.blobTracker.regionMap.decMaxTrackMove0();
			System.out.println("maxTrackMove0 decreased to " + blobPanel.blobTracker.regionMap.getMaxTrackMove0());
			break;
		case KeyEvent.VK_Y:
			blobPanel.blobTracker.regionMap.incMaxTrackMove0();
			System.out.println("maxTrackMove0 increased to " + blobPanel.blobTracker.regionMap.getMaxTrackMove0());
			break;//
		case KeyEvent.VK_I:
			blobPanel.blobTracker.tracker.csp.setNonMatchPenalty(blobPanel.blobTracker.tracker.csp.getNonMatchPenalty() + 1);
			System.out.println("nonMatchPenalty  increased to " +blobPanel.blobTracker.tracker.csp.getNonMatchPenalty());
			props.setProperty("nonMatchPenalty", blobPanel.blobTracker.tracker.csp.getNonMatchPenalty());			
			break;
		case KeyEvent.VK_U:
			blobPanel.blobTracker.tracker.csp.setNonMatchPenalty(blobPanel.blobTracker.tracker.csp.getNonMatchPenalty() - 1);
			System.out.println("nonMatchPenalty  decreased to " +blobPanel.blobTracker.tracker.csp.getNonMatchPenalty());
			props.setProperty("nonMatchPenalty", blobPanel.blobTracker.tracker.csp.getNonMatchPenalty());			
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
			props.store();
			break;	
		case '.':
			blobPanel.blobTracker.presenceDetector.nextMode();
			break;
		case ',':
			blobPanel.blobTracker.presenceDetector.prevMode();
			break;
		case 'r':
			blobPanel.blobTracker.presenceDetector.resetBackground(3);
			break;

			
		}
	}

	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}
*/
}
