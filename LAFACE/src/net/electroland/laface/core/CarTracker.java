package net.electroland.laface.core;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTrackerServer;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;

public class CarTracker extends Thread implements TrackListener{
	LinkedBlockingQueue<TrackResults> resultsQueue;
	ConcurrentHashMap<Integer,Track> cars;
	BlobTrackerServer bts;
	LAFACEMain main;
	
	public CarTracker(LAFACEMain main){
		this.main = main;
		resultsQueue = new LinkedBlockingQueue<TrackResults>();		// used to get info on active blobs
		cars = new ConcurrentHashMap<Integer,Track>();				// stores active blob data (position history, speed, etc)
		ElProps.init("depends//blobTracker.props");
		bts = new BlobTrackerServer(ElProps.THE_PROPS);
		bts.addTrackListener(0, this);
	}

	public void run(){
		while(true) {
			try {
				
				TrackResults result = resultsQueue.take(); // will block until something is on the que
				//System.out.println(result.created.size() + " tracks created");
				if(result.created.size() > 0){
					Iterator<Track> iter = result.created.iterator();
					while(iter.hasNext()){
						Track newtrack = iter.next();
						// REMEMBER: the camera is mirroring the display
						if(newtrack.x < Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString())/2){
							Impulse impulse = new Impulse(main, 0, 300, false);	// left side
							impulse.start();
						} else {
							Impulse impulse = new Impulse(main, 0, 300, true);		// right side
							impulse.start();
						}
					}
				}
				
				if(result.existing.size() > 0){
					// set damping based on population size
					Wave wave = ((WaveShow)(main.getCurrentAnimation())).getWaves().get(0);	// set specifically for single wave sprite instance
					// TODO set a min/max damping value in properties based on population sizes
					if(result.existing.size() < 9){
						wave.setDamping(result.existing.size()/40);
					} else {
						wave.setDamping(0.2);	// max damping cap
					}
				} else {
					// set damping to 0
					Wave wave = ((WaveShow)(main.getCurrentAnimation())).getWaves().get(0);	// set specifically for single wave sprite instance
					wave.setDamping(0);	// allows the existing wave action to continue infinitely if no new traffic comes
				}
				//System.out.println(result.existing.size() + " tracks");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				sleep(33);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public void addTrackListener(TrackListener listener){
		bts.addTrackListener(0, listener);
	}
	
	public void updateTracks(TrackResults results) {
		resultsQueue.offer(results);	
	}

}
