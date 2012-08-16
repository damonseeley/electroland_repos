package net.electroland.elvis.blobktracking.core;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import net.electroland.elvis.blobtracking.Tracker;
import net.electroland.elvis.imaging.Filter;
import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.Parameter;

// provides standard key controls for all UIs using presence detector
public class PresenceDetectorKeyListener implements KeyListener {
	ElProps props;
	PresenceDetector presenceDetector;
	Tracker tracker;

	public PresenceDetectorKeyListener(ElProps props, PresenceDetector pd) {
		presenceDetector = pd;
		tracker = pd.tracker;
		this.props = props;
	}
	public void keyReleased(KeyEvent e) {
		if(e.getKeyChar() == '?') {
			System.out.println("? - this help menu");
			System.out.println("s - save current properties (will overwrite existing file)");
			System.out.println(". - view next window mode");
			System.out.println(", - view prev window mode");
			System.out.println("m/n - increase/decrease threshold");
			System.out.println("x/z - increase/decrease background adaption rate");
			System.out.println("p/o - increase/decrease provisionalPenalty for track");
			System.out.println("i/u - increase/decrease nonMatchPenalty for track");
			System.out.println("y/t - increase/decrease max track size");
			System.out.println("h/g - increase/decrease min track size");
			System.out.println("r - reset background model");
		}
		
		switch(e.getKeyCode()) {
		case KeyEvent.VK_S:
			props.store();
			break;
		case KeyEvent.VK_PERIOD:
			presenceDetector.nextMode();
			break;
		case KeyEvent.VK_COMMA:
			presenceDetector.prevMode();
			break;			
		case KeyEvent.VK_R:
			presenceDetector.resetBackground(3);
			break;



		}
	}
	
	public void keyPressed(KeyEvent e) {
		Filter curFilter = presenceDetector.getCurrentFilter();
		if(curFilter == null) return; // nothing to set
		boolean isInc = true;
		int curParam = -1;
		switch(e.getKeyCode()) {
		case KeyEvent.VK_MINUS:
			isInc = false;
		case KeyEvent.VK_EQUALS:
			curParam = 0;
			break;
		case KeyEvent.VK_9:
			isInc = false;
		case KeyEvent.VK_0:
			curParam = 1;
			break;			
		case KeyEvent.VK_7:
			isInc = false;
		case KeyEvent.VK_8:
			curParam = 2;
			break;
		case KeyEvent.VK_5:
			isInc = false;
		case KeyEvent.VK_6:
			curParam = 3;
			break;
		case KeyEvent.VK_3:
			isInc = false;
		case KeyEvent.VK_4:
			curParam = 4;
			break;
		case KeyEvent.VK_1:
			isInc = false;
		case KeyEvent.VK_2:
			curParam = 5;
			break;
		case KeyEvent.VK_OPEN_BRACKET:
			isInc = false;
		case KeyEvent.VK_CLOSE_BRACKET:
			curParam = 6;
			break;
		}
		if(curParam != -1) {
			if(isInc) {
				curFilter.incParameter(curParam);
				Parameter p = curFilter.getParameter(curParam);
				if(p != null) {
					System.out.println(p.getName() + " increased to " +p.getDoubleValue());
				p.writeToProps(props);
				}
	
			} else {
				curFilter.decParameter(curParam);
				Parameter p = curFilter.getParameter(curParam);
				if(p != null) {
					System.out.println(p.getName() + " increased to " +p.getDoubleValue());
				p.writeToProps(props);
				}
			}
		}
		/*
		case KeyEvent.VK_M:
			presenceDetector.setThresh(presenceDetector.getThresh() + 1);
			System.out.println("theshold increased to " + presenceDetector.getThresh() );
			props.setProperty("threshold", presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_N:
			presenceDetector.setThresh(presenceDetector.getThresh() -1);
			System.out.println("theshold decreased to " + presenceDetector.getThresh() );
			props.setProperty("threshold",presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_X:
			presenceDetector.setThresh(presenceDetector.getAdaptation() + .0001);
			System.out.println("adaptation increased to " + presenceDetector.getAdaptation() );
			props.setProperty("adaptation",presenceDetector.getAdaptation() );			
			break;
		case KeyEvent.VK_Z:
			presenceDetector.setThresh(presenceDetector.getAdaptation() - .0001);
			System.out.println("adaptation adaptation decreased to " +presenceDetector.getAdaptation());
			props.setProperty("adaptation", presenceDetector.getAdaptation());			
			break;

			// ds additions	
		case KeyEvent.VK_P:
			if(tracker != null) {
				tracker.csp.setProvisionalPenalty(tracker.csp.getProvisionalPenalty() + 1);
				System.out.println("provisionalPenalty  increased to " +tracker.csp.getProvisionalPenalty());
				props.setProperty("provisionalPenalty", tracker.csp.getProvisionalPenalty());			
			}
			break;
		case KeyEvent.VK_O:
			if(tracker != null) {
				tracker.csp.setProvisionalPenalty(tracker.csp.getProvisionalPenalty() - 1);
				System.out.println("provisionalPenalty  decreased to " +tracker.csp.getProvisionalPenalty());
				props.setProperty("provisionalPenalty", tracker.csp.getProvisionalPenalty());			
			}
			break;
			/* these are not easy to change while running
		case KeyEvent.VK_T:
			blobTracker.regionMap.decMaxTrackMove0();
			System.out.println("maxTrackMove0 decreased to " + blobTracker.regionMap.getMaxTrackMove0());
			break;
		case KeyEvent.VK_Y:
			blobTracker.regionMap.incMaxTrackMove0();
			System.out.println("maxTrackMove0 increased to " + blobTracker.regionMap.getMaxTrackMove0());
			break;*/
		/*
		case KeyEvent.VK_I:
			if(tracker == null) return;
			tracker.csp.setNonMatchPenalty(tracker.csp.getNonMatchPenalty() + 1);
			System.out.println("nonMatchPenalty  increased to " +tracker.csp.getNonMatchPenalty());
			props.setProperty("nonMatchPenalty", tracker.csp.getNonMatchPenalty());			
			break;
		case KeyEvent.VK_U:
			if(tracker == null) return;
			tracker.csp.setNonMatchPenalty(tracker.csp.getNonMatchPenalty() - 1);
			System.out.println("nonMatchPenalty  decreased to " +tracker.csp.getNonMatchPenalty());
			props.setProperty("nonMatchPenalty", tracker.csp.getNonMatchPenalty());			
			break;
		case KeyEvent.VK_Y:
			presenceDetector.setMaxBlobSize(presenceDetector.getMaxBlobSize() + 1);
			System.out.println("maxBlobSize increased to " +presenceDetector.getMaxBlobSize());
			props.setProperty("maxBlobSize", presenceDetector.getMaxBlobSize());			
			break;
		case KeyEvent.VK_T:
			presenceDetector.setMaxBlobSize(presenceDetector.getMaxBlobSize() - 1);
			System.out.println("maxBlobSize decreased to " +presenceDetector.getMaxBlobSize());
			props.setProperty("maxBlobSize", presenceDetector.getMaxBlobSize());			
			break;
		case KeyEvent.VK_H:
			presenceDetector.setMinBlobSize(presenceDetector.getMinBlobSize() + 1);
			System.out.println("minBlobSize increased to " +presenceDetector.getMinBlobSize());
			props.setProperty("minBlobSize", presenceDetector.getMinBlobSize());			
			break;
		case KeyEvent.VK_G:
			presenceDetector.setMinBlobSize(presenceDetector.getMinBlobSize() - 1);
			System.out.println("minBlobSize increased to " +presenceDetector.getMinBlobSize());
			props.setProperty("minBlobSize", presenceDetector.getMinBlobSize());			
			break;
		}	
		*/
	}



	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub

	}

}
