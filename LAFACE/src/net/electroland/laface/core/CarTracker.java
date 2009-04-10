package net.electroland.laface.core;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTrackerServer;
import net.electroland.blobTracker.util.ElProps;

public class CarTracker extends Thread implements TrackListener{
	LinkedBlockingQueue<TrackResults> resultsQueue;
	ConcurrentHashMap<Integer,Track> cars;
	BlobTrackerServer bts;
	
	public CarTracker(){
		resultsQueue = new LinkedBlockingQueue<TrackResults>();		// used to get info on active blobs
		cars = new ConcurrentHashMap<Integer,Track>();				// stores active blob data (position history, speed, etc)
		ElProps.init("depends//blobTracker.props");
		bts = new BlobTrackerServer(ElProps.THE_PROPS);
		bts.addTrackListener(0, this);
	}

	public void run(){
		while(true) {
			/*
			try {
				
				TrackResults result = resultsQueue.take(); // will block until something is on the que
				System.out.println(result.created.size() + " tracks created");
				if(result.created.size() > 0){
					Iterator<Track> iter = result.created.iterator();
					while(iter.hasNext()){
						Track newtrack = iter.next();
						
					}
				}
				//System.out.println(result.existing.size() + " tracks");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
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
