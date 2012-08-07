package net.electroland.elvis.blobktracking.core;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.IOException;

import net.electroland.elvis.blobktracking.ui.BlobFrame;
import net.electroland.elvis.blobktracking.ui.Console;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.manager.ImagePanel;
import net.electroland.elvis.util.ElProps;

public class BlobTrackerServer implements KeyListener {

	BlobTracker blobTracker;
	ElProps props;

	public BlobTrackerServer(ElProps props) { 




		if(props.getProperty("showConsole", false)) {
			Console console  = new Console("");
			console.setVisible(true);
			console.addKeyListener(this);
		} 

		blobTracker = new BlobTracker(props.getProperty("srcWidth", 640), props.getProperty("srcHeight", 480));

		if(props.getProperty("showGraphics", true)) {
			BlobFrame bf = new BlobFrame("el blob", blobTracker);
			bf.blobPanel.addKeyListener(this);

		}

		blobTracker.presenceDetector.setThresh(ElProps.THE_PROPS.getProperty("threshold", 100.0));
		blobTracker.presenceDetector.setAdaptation(ElProps.THE_PROPS.setProperty("adaptation", .001));

		try {
//			blobTracker.setSourceStream(props.getProperty("camera", ImagePanel.FLY_SRC));
			blobTracker.setSourceStream(props.getProperty("camera", ImagePanel.LOCALAXIS_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}


		blobTracker.presenceDetector.start();




	}

	public static void main(String arg[]) throws IOException {
		if(arg.length > 0) {
			ElProps.init(arg[0]);
		} else {
			ElProps.init("blobTracker.props");
		}

		new BlobTrackerServer(
				ElProps.THE_PROPS
		);




	}

	public void keyPressed(KeyEvent e) {
		switch(e.getKeyCode()) {

		case KeyEvent.VK_M:
			blobTracker.presenceDetector.setThresh(blobTracker.presenceDetector.getThresh() + 1);
			System.out.println("theshold increased to " + blobTracker.presenceDetector.getThresh() );
			ElProps.THE_PROPS.setProperty("threshold",blobTracker.presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_N:
			blobTracker.presenceDetector.setThresh(blobTracker.presenceDetector.getThresh() -1);
			System.out.println("theshold increased to " + blobTracker.presenceDetector.getThresh() );
			ElProps.THE_PROPS.setProperty("threshold",blobTracker.presenceDetector.getThresh() );			
			break;
		case KeyEvent.VK_X:
			blobTracker.presenceDetector.setThresh(blobTracker.presenceDetector.getAdaptation() + .0001);
			System.out.println("adaptation increased to " + blobTracker.presenceDetector.getAdaptation() );
			ElProps.THE_PROPS.setProperty("adaptation",blobTracker.presenceDetector.getAdaptation() );			
		break;
		case KeyEvent.VK_Z:
			blobTracker.presenceDetector.setThresh(blobTracker.presenceDetector.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " +blobTracker.presenceDetector.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", blobTracker.presenceDetector.getAdaptation());			
			break;
			
		// ds additions	
		case KeyEvent.VK_P:
			blobTracker.tracker.csp.setProvisionalPenalty(blobTracker.tracker.csp.getProvisionalPenalty() + 1);
			System.out.println("provisionalPenalty  increased to " +blobTracker.tracker.csp.getProvisionalPenalty());
			ElProps.THE_PROPS.setProperty("provisionalPenalty", blobTracker.tracker.csp.getProvisionalPenalty());			
			break;
		case KeyEvent.VK_O:
			blobTracker.tracker.csp.setProvisionalPenalty(blobTracker.tracker.csp.getProvisionalPenalty() - 1);
			System.out.println("provisionalPenalty  decreased to " +blobTracker.tracker.csp.getProvisionalPenalty());
			ElProps.THE_PROPS.setProperty("provisionalPenalty", blobTracker.tracker.csp.getProvisionalPenalty());			
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
		case KeyEvent.VK_I:
			blobTracker.tracker.csp.setNonMatchPenalty(blobTracker.tracker.csp.getNonMatchPenalty() + 1);
			System.out.println("nonMatchPenalty  increased to " +blobTracker.tracker.csp.getNonMatchPenalty());
			ElProps.THE_PROPS.setProperty("nonMatchPenalty", blobTracker.tracker.csp.getNonMatchPenalty());			
			break;
		case KeyEvent.VK_U:
			blobTracker.tracker.csp.setNonMatchPenalty(blobTracker.tracker.csp.getNonMatchPenalty() - 1);
			System.out.println("nonMatchPenalty  decreased to " +blobTracker.tracker.csp.getNonMatchPenalty());
			ElProps.THE_PROPS.setProperty("nonMatchPenalty", blobTracker.tracker.csp.getNonMatchPenalty());			
			break;
		}	
	}

	public void addTrackListener( TrackListener tl) {
		blobTracker.tracker.addListener(tl);
	}
	
	public void removeTrackListener(TrackListener tl) {
		blobTracker.tracker.removeListener(tl);
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
			ElProps.THE_PROPS.store();
			break;
		case KeyEvent.VK_PERIOD:
			blobTracker.presenceDetector.nextMode();
			break;
		case KeyEvent.VK_COMMA:
			blobTracker.presenceDetector.prevMode();
			break;			
		case KeyEvent.VK_R:
			blobTracker.presenceDetector.resetBackground(3);
			break;



		}
	}

	public void keyTyped(KeyEvent e) {

	}


}
