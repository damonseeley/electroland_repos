package net.electroland.laface.tracking;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTrackerServer;
import net.electroland.blobTracker.util.ElProps;

/**
 * Measures track locations and creates a Mover object with location, velocity,
 * and acceleration / deceleration values to be used as an interpretive layer between
 * blob tracking and sprite drawing.
 * 
 * @author Aaron Siegel
 */

public class Tracker extends Thread implements TrackListener {
	
	private int sampleSize;
	private ConcurrentHashMap<Integer,Mover> movers;
	private ConcurrentHashMap<Integer,Candidate> candidates;
	private LinkedBlockingQueue<TrackResults> resultsQueue;
	private BlobTrackerServer bts;
	private boolean running = true;
	
	public Tracker(int sampleSize){
		this.sampleSize = sampleSize;								// minimum number of location/time samples to create a mover
		movers = new ConcurrentHashMap<Integer,Mover>();
		candidates = new ConcurrentHashMap<Integer,Candidate>();
		resultsQueue = new LinkedBlockingQueue<TrackResults>();		// used to get info on active tracks
		ElProps.init("depends//blobTracker.props");					// load tracking properties
		bts = new BlobTrackerServer(ElProps.THE_PROPS);				// launch the blob tracker
		bts.addTrackListener(0, this);
	}
	
	public void run(){
		while(running){
			try {
				TrackResults result = resultsQueue.take();			// will block until something is on the queue
				if(result.created.size() > 0){						// CREATED
					Iterator<Track> iter = result.created.iterator();
					while(iter.hasNext()){
						Track newtrack = iter.next();
						candidates.put(newtrack.id, new Candidate());
					}
				} else if(result.existing.size() > 0){				// EXISTING
					Iterator<Track> iter = result.existing.iterator();
					while(iter.hasNext()){
						Track track = iter.next();
						Candidate candidate = candidates.get(track.id);		// get the candidate that matches this track
						candidate.addLocation(track.x, track.y);			// add the new location to history...
						if(candidate.getLocations().size() == sampleSize){	// if it meets the sample size...
							movers.put(track.id, new Mover(candidate));		// make a mover from this candidate
							candidates.remove(track.id);					// remove candidate
						}
					}
				}
//				} else if(result.deleted.size() > 0){				// DELETED
//					Iterator<Track> iter = result.deleted.iterator();
//					while(iter.hasNext()){
//						Track deadtrack = iter.next();
//						if(movers.containsKey(deadtrack.id)){
//							movers.remove(deadtrack.id);
//						}
//					}
//				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
	}
	
	public void updateTracks(TrackResults results) {
		resultsQueue.offer(results);
	}

}
