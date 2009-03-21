package net.electroland.blobTracker.core;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.IOException;

import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobTracker.ui.BlobFrame;
import net.electroland.blobTracker.ui.Console;
import net.electroland.blobTracker.util.ElProps;

public class BlobTrackerServer implements KeyListener {

	BlobTracker blobTracker;
	ElProps props;

	public BlobTrackerServer(ElProps props) { 

		this.props = props;



		if(props.getProperty("showConsole", false)) {
			Console console  = new Console("");
			console.setVisible(true);
			console.addKeyListener(this);
		} 

		blobTracker = new BlobTracker(props.getProperty("srcWidth", 320), props.getProperty("srcHeight", 240));

		if(props.getProperty("showGraphics", true)) {
			BlobFrame bf = new BlobFrame("el blob", blobTracker);
			bf.blobPanel.addKeyListener(this);

		}

		blobTracker.setThresh(ElProps.THE_PROPS.getProperty("threshold", 5000.0));
		blobTracker.setBackgroundAdaptation(ElProps.THE_PROPS.setProperty("adaptation", .001));

		try {
			blobTracker.setSourceStream(props.getProperty("camera", BlobTracker.LOCALAXIS_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}


		blobTracker.start();




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
			blobTracker.setThresh(blobTracker.getThresh() + 100);
			System.out.println("theshold increased to " +blobTracker.getThresh());
			ElProps.THE_PROPS.setProperty("threshold", blobTracker.getThresh());			
			break;
		case KeyEvent.VK_N:
			blobTracker.setThresh(blobTracker.getThresh() - 100);
			System.out.println("theshold decreased to " + blobTracker.getThresh());
			ElProps.THE_PROPS.setProperty("threshold", blobTracker.getThresh());
			break;
		case KeyEvent.VK_X:
			blobTracker.setBackgroundAdaptation(blobTracker.getAdaptation() + .0001);
			System.out.println("background adaptation increased to " + blobTracker.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", blobTracker.getAdaptation());			
			break;
		case KeyEvent.VK_Z:
			blobTracker.setBackgroundAdaptation(blobTracker.getAdaptation() - .0001);
			System.out.println("background adaptation decreased to " + blobTracker.getAdaptation());
			ElProps.THE_PROPS.setProperty("adaptation", blobTracker.getAdaptation());			
			break;
		}
	}

	public void addTrackListener(int regionID, TrackListener tl) {
		blobTracker.tracker[regionID].addListener(tl);
	}
	
	public void removeTrackListener(int regionID, TrackListener tl) {
		blobTracker.tracker[regionID].removeListener(tl);
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
			blobTracker.nextMode();
			break;
		case KeyEvent.VK_COMMA:
			blobTracker.prevMode();
			break;			
		case KeyEvent.VK_R:
			blobTracker.resetBackground(3);
			break;



		}
	}

	public void keyTyped(KeyEvent e) {

	}


}
