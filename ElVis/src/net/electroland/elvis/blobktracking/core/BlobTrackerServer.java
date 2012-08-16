package net.electroland.elvis.blobktracking.core;

import java.awt.event.WindowEvent;
import java.io.IOException;

import net.electroland.elvis.blobktracking.ui.BlobFrame;
import net.electroland.elvis.blobktracking.ui.Console;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.manager.ImagePanel;
import net.electroland.elvis.util.ElProps;

public class BlobTrackerServer {

	BlobTracker blobTracker;
	ElProps props;

	public BlobTrackerServer(ElProps props) { 

		this.props = props;

		Console console = null;
		if(props.getProperty("showConsole", false)) {
			console  = new Console("");
			console.setVisible(true);
		} 

		blobTracker = new BlobTracker(props);

		if(props.getProperty("showGraphics", true)) {
			BlobFrame bf = new BlobFrame(props, "el blob", blobTracker);
			bf.blobPanel.addKeyListener(new PresenceDetectorKeyListener(props,blobTracker.presenceDetector));
			bf.addWindowListener(new java.awt.event.WindowAdapter() {
				public void windowClosing(WindowEvent winEvt) {
					blobTracker.stopRunning();
				}
			});


		}
		if(console != null) {
		console.addKeyListener(new PresenceDetectorKeyListener(props,blobTracker.presenceDetector));
		}
		blobTracker.presenceDetector.thresh.getParameter(0).setValue(props.getProperty("threshold", 100.0));
		blobTracker.presenceDetector.background.getParameter(0).setValue(props.getProperty("adaptation", .001));

		try {
//			blobTracker.setSourceStream(props.getProperty("camera", ImagePanel.FLY_SRC));
			blobTracker.setSourceStream(props.getProperty("camera", ImagePanel.LOCALAXIS_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}


		blobTracker.presenceDetector.start();




	}

	



		



	public void addTrackListener( TrackListener tl) {
		blobTracker.tracker.addListener(tl);
	}
	
	public void removeTrackListener(TrackListener tl) {
		blobTracker.tracker.removeListener(tl);
	}



	public static void main(String arg[]) throws IOException {
		ElProps props;
		if(arg.length > 0) {
			props = ElProps.init(arg[0]);
		} else {
			props =ElProps.init("blobTracker.props");
		}

		new BlobTrackerServer(
				props
		);
	}
}
