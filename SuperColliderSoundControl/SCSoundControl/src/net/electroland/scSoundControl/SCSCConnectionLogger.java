package net.electroland.scSoundControl;

import org.apache.log4j.Logger;


public class SCSCConnectionLogger implements SCSoundControlNotifiable {

	boolean serverIsLive = false;
	
	private static Logger logger = Logger.getLogger(SCSCConnectionLogger.class);

	public void setup() {
		
	}
	

	//unneeded, but required by notifiable interface
	public void receiveNotification_BufferLoaded(int id, String filename) {}

	//go live
	public void receiveNotification_ServerRunning() {
		serverIsLive = true;
		logger.info("SCSynth online");
	}

	//get server load update
	public void receiveNotification_ServerStatus(float averageCPU, float peakCPU, int numCurSynths) {
	
	}

	//not live
	public void receiveNotification_ServerStopped() {
		serverIsLive = false;
		logger.info("SCSynth offline");
	}
	
	public void notify_currentPolyphony(float polyphony) {

	}
	
}
