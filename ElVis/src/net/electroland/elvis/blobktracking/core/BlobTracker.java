package net.electroland.elvis.blobktracking.core;

import java.io.IOException;
import java.util.Vector;

import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.blobtracking.Track;
import net.electroland.elvis.blobtracking.Tracker;
import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.ImageAcquirer;
import net.electroland.elvis.imaging.acquisition.axisCamera.FlowerCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.LocalCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NavyCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoSouthCam;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.elvis.imaging.acquisition.openCV.FlyCamera;
import net.electroland.elvis.manager.ImagePanel;
import net.electroland.elvis.util.CameraFactory;
import net.electroland.elvis.util.ElProps;

import com.googlecode.javacv.FrameGrabber.Exception;

public class BlobTracker {
	
	public Tracker tracker;
	
	public PresenceDetector presenceDetector;
	ImageAcquirer srcStream;
	
	public Vector<Blob> newFrameBlobs = new Vector<Blob>();
	
	public BlobTracker(ElProps props) {
		presenceDetector = new PresenceDetector(props, true);
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
			srcStream = null;
		}
		int frameSkip = 2;
		try {
			srcStream = CameraFactory.camera(s, presenceDetector.getSrcWidth(),presenceDetector.getSrcHeight() , presenceDetector);
		} catch (Exception e) {
			e.printStackTrace();
			srcStream=null;
		}
		if(srcStream == null) {
			throw new IOException("Unknown or invalid source");
		}
		srcStream.start();
		presenceDetector.resetBackground(frameSkip);
		
	}
	
	public Vector<Track> getTracks() {
		return presenceDetector.tracker.tracks;
	}



	

}
