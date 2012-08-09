package net.electroland.elvis.example;

import java.io.File;
import java.util.Vector;

import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.axisCamera.NoHoNorthCam;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.util.ElProps;

public class ExampleNoHoNorth extends Thread {
	public boolean isRunning = true;
	NoHoNorthCam cam;
//	WebCam cam;
	PresenceDetector detector;
	
	public ExampleNoHoNorth() {
		detector = PresenceDetector.createFromFile(new ElProps(), new File("noho_north.elv"));
		//cam = new LocalCam(160,120,detector);

		cam = new NoHoNorthCam(160,120, detector, false);

	}
public void stopRunning() {
	isRunning = false;
}
	public void run() {
		cam.start();
		detector.start(); // start detector before cam or else you'll get synchronization problems
		Vector<PolyRegion> regions = detector.getRegions();
		boolean[] triggers = new boolean[regions.size()];
		
		// init to false which should be done by java anyway
		for(int i =0; i < triggers.length; i++) {
			triggers[i] = false;
		}
		
		

		while(isRunning) {
			int i = 0;
			for(PolyRegion region : regions) { // check all the regions
//				 we are only going to print stuff out when state changes
				if(region.isTriggered) {  // if triggered now
					if(! triggers[i]) { // but not previously
						System.out.println(region.name + " is triggered");
						triggers[i] = true;
					}
				} else { // if not triggered now
					if(triggers[i]) { // but was previosly
						System.out.println(region.name + " is no longer triggered");
						triggers[i] = false;
					}					
				}
				i++;

			}
			// sleep a bit so we don't hammer the processor
			try { sleep(100); } catch (InterruptedException e) {};
		}
		cam.stopRunning();
		detector.stopRunning();
		
	}
	
	public static void main(String args[]) {
		new ExampleNoHoNorth().start();
	}
}
