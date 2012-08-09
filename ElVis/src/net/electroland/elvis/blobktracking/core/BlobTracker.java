package net.electroland.elvis.blobktracking.core;

import java.io.IOException;
import java.util.Vector;

import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.blobtracking.Track;
import net.electroland.elvis.blobtracking.Tracker;
import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.FlyCapture.FlyCamera;
import net.electroland.elvis.imaging.acquisition.axisCamera.FlowerCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.LocalCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NavyCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoSouthCam;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvis.manager.ImagePanel;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.FrameGrabber.Exception;

public class BlobTracker {
	int w;
	int h;
	public Tracker tracker;
	
	public PresenceDetector presenceDetector;
	ImageAcquirer srcStream;
	
	public Vector<Blob> newFrameBlobs = new Vector<Blob>();
	
	public BlobTracker(ElProps props) {
		w = props.getProperty("srcWidth", 640); 
		h = props.getProperty("srcHeight", 480);
		presenceDetector = new PresenceDetector(props, w, h, true);
		tracker = presenceDetector.tracker;

	}
	
	public void receiveErrorMsg(Exception cameraException) {
		// TODO Auto-generated method stub
		
	}

	public void stopRunning() {
		presenceDetector.stopRunning();
		try {
			Thread.sleep(100); // wait for image to be added to queue so will exit
		} catch (InterruptedException e) {
		} 
		if(srcStream != null) {
			srcStream.stopRunning();
		}
		
	}
	
	public void setSourceStream(String s) throws IOException {
		if(srcStream != null) {
			srcStream.stopRunning();
		}
		int frameSkip = 2;
		if(s.equals(ImagePanel.NAVY_SRC)) {
			srcStream = new NavyCam(w,h,presenceDetector, false);
		} else if(s.equals(ImagePanel.FLOWER_SRC)) {
			srcStream = new FlowerCam(w,h,presenceDetector, false);
		} else if(s.equals(ImagePanel.NOHOSOUTH_SRC)) {
				srcStream = new NoHoSouthCam(w,h,presenceDetector, false);
		} else if(s.equals(ImagePanel.NOHONORTH_SRC)) {
			srcStream = new NoHoNorthCam(w,h,presenceDetector, false);
		} else if(s.equals(ImagePanel.JMYRON_SRC)) {
			frameSkip = 50;
			srcStream = new WebCam(w, h, 12, presenceDetector, false);
		} else if(s.equals(ImagePanel.LOCALAXIS_SRC)) {
			System.out.println("creating local " + w +"x" +h);
			srcStream = new LocalCam(w,h,presenceDetector, false);
		} else if(s.equals(ImagePanel.FLY_SRC)) {
			System.out.println("creating fly camera " + w +"x" +h);
			try {
				srcStream = new FlyCamera(presenceDetector, 0 , w, h);
			} catch (Exception e) {
				srcStream = null;
				e.printStackTrace();
			}
		}else {
			srcStream = null;
			throw new IOException("Unknown source");
		}
		srcStream.start();
		presenceDetector.resetBackground(frameSkip);
	}
	
	public Vector<Track> getTracks() {
		return presenceDetector.tracker.tracks;
	}



	

}
