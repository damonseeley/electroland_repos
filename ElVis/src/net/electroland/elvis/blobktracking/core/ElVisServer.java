package net.electroland.elvis.blobktracking.core;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.blobktracking.ui.BlobFrame;
import net.electroland.elvis.blobktracking.ui.Console;
import net.electroland.elvis.blobktracking.ui.PresenceDetectorKeyListener;
import net.electroland.elvis.blobtracking.BlobTracker;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.imaging.PresenceDetector.ImgReturnType;
import net.electroland.elvis.net.ImageClient;
import net.electroland.elvis.net.ImageServer;
import net.electroland.elvis.net.TrackUDPBroadcaster;
import net.electroland.elvis.util.CameraFactory;
import net.electroland.elvis.util.ElProps;

public class ElVisServer {

	BlobTracker blobTracker;
	ElProps props;
	TrackUDPBroadcaster trackBrodcaster;
	ImageServer imageServer;

	public ElVisServer(ElProps props) throws SocketException, UnknownHostException { 
		
		this.props = props;

		Console console = null;
		if(props.getProperty("showConsole", false)) {
			console  = new Console("");
			console.setVisible(true);
		}
		

		blobTracker = new BlobTracker(props);
/*		
		if(props.getProperty("broadcastTracks", false)) {
			String address = props.getProperty("broadcastTracksAddress", "localhost");
			int port = props.getProperty("broadcastTracksPort", 3789);
			try {
				trackBrodcaster = new TrackUPDBroadcaster(address, port);
				trackBrodcaster.start();
				blobTracker.tracker.addListener(trackBrodcaster);
			} catch (SocketException e) {
				e.printStackTrace();
			} catch (UnknownHostException e) {
				e.printStackTrace();
			}
		}
*/
		
		if(props.getProperty("showGraphics", true)) {
			BlobFrame bf = new BlobFrame(props, "el blob", blobTracker);
			bf.blobPanel.addKeyListener(new PresenceDetectorKeyListener(props,blobTracker.presenceDetector));
			bf.addWindowListener(new java.awt.event.WindowAdapter() {
				public void windowClosing(WindowEvent winEvt) {
					blobTracker.stopRunning();
					System.exit(0); //calling the method is a must
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
			blobTracker.setSourceStream(props.getProperty("camera", CameraFactory.FLY_SRC));
		} catch (IOException e) {
			e.printStackTrace();
		}


		blobTracker.presenceDetector.start();
		
		if(props.getProperty("runImageServer", true)) {
			try {
				imageServer= new ImageServer(blobTracker.presenceDetector, props.getProperty("imageServerPort", 3598));
				imageServer.start();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		//temp
		
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
			props =ElProps.init("depends/blobTracker.props");
		}

		new ElVisServer(
				props
				);
	}
}
