package net.electroland.elvis.blobktracking.core;

import java.awt.event.WindowEvent;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.blobktracking.ui.BlobFrame;
import net.electroland.elvis.blobktracking.ui.Console;
import net.electroland.elvis.blobktracking.ui.PresenceDetectorKeyListener;
import net.electroland.elvis.blobtracking.BlobTracker;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.imaging.acquisition.openCV.OpenCVCam;
import net.electroland.elvis.net.ImageServer;
import net.electroland.elvis.net.TrackUDPBroadcaster;
import net.electroland.elvis.util.CameraFactory;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.TimeOutMonitorThread;
import net.electroland.elvis.util.TimeOutMonitorThread.TimeOutListener;

import org.apache.log4j.Logger;

public class ElVisServer implements TimeOutListener {
	public static Logger logger = Logger.getLogger("com.electroland.ElVis");
	BlobTracker blobTracker;
	ElProps props;
	TrackUDPBroadcaster trackBrodcaster;
	ImageServer imageServer;

	public ElVisServer(ElProps props) throws SocketException, UnknownHostException { 		
		logger.info("ElVis starting up with property file " + props.fileName);
		this.props = props;

		Console console = null;
		if(props.getProperty("showConsole", false)) {
			console  = new Console("");
			console.setVisible(true);
            console.addWindowListener(new java.awt.event.WindowAdapter() {
                public void windowClosing(WindowEvent winEvt) {
                    blobTracker.stopRunning();
                    System.exit(0); //calling the method is a must
                }
            });
			
		} else {
			logger.info("not showing console, showConsole is false");
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
			bf.setPolyRegions(blobTracker.presenceDetector.getRegions());
			bf.blobPanel.addKeyListener(new PresenceDetectorKeyListener(props,blobTracker.presenceDetector));
			bf.addWindowListener(new java.awt.event.WindowAdapter() {
				public void windowClosing(WindowEvent winEvt) {
					blobTracker.stopRunning();
					System.exit(0); //calling the method is a must
				}
			});


		} else {
			logger.info("not showing GUI, showGraphics is false");
		}
		if(console != null) {
			console.addKeyListener(new PresenceDetectorKeyListener(props,blobTracker.presenceDetector));
		}
		blobTracker.presenceDetector.thresh.getParameter(0).setValue(props.getProperty("threshold", 100.0));
		blobTracker.presenceDetector.background.getParameter(0).setValue(props.getProperty("adaptation", .001));


		setupCamera();
		
		blobTracker.presenceDetector.start();

		if(props.getProperty("runImageServer", true)) {
			try {
				imageServer= new ImageServer(blobTracker.presenceDetector, props.getProperty("imageServerPort", 3598));
				imageServer.start();
			} catch (IOException e) {
				logger.error(e.toString());
				e.printStackTrace();
			}
		}


	}
	protected void setupCamera() {
		try {

			String cameraSrcString = props.getProperty("camera", CameraFactory.FLY_SRC);
			blobTracker.setSourceStream(cameraSrcString);
			if(cameraSrcString.equals(CameraFactory.FLY_SRC)) {
				long cameraTimeOutDuration = props.getProperty("cameraTimeOutDuration", 2000);
				if(cameraTimeOutDuration > 0) {
					logger.info("camera timeout duration " + cameraTimeOutDuration);
					TimeOutMonitorThread tom = new TimeOutMonitorThread(cameraTimeOutDuration);
					tom.addTimeOutListener(this);
					((OpenCVCam) blobTracker.getSourceStream()).setTimeOutMonitorThread(tom);
					tom.start();
				} else {
					logger.info("not monitoring camera time outs, cameraTimeOutDuration <= 0");
				}
			}


		} catch (IOException e) {
			logger.error(e.toString());
			e.printStackTrace();
		}
		
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
			props =ElProps.init("elvis.properties");
		}

		new ElVisServer(
				props
				);
	}











	@Override
	public void monitorTimedOut(TimeOutMonitorThread sender) {
		logger.error("Camera timed out");
		if(props.getProperty("cameraTimeOutRestart", false)) {
			// replace everything in this if statement with exit(0)
			logger.warn("attempting to restart camera - stopping current threads");
			sender.stopRunning();
			blobTracker.getSourceStream().stopRunningForce();
			long delay = props.getProperty("cameraTimeOutWait", 5000);
			logger.info("waiting " +delay + " to restart camera");
			synchronized(this) {
				try {
					wait(delay);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			logger.info("attempting to restart camera");
			setupCamera();
		} else {
			logger.warn("camera restart not requested");			
		}

	}
}
