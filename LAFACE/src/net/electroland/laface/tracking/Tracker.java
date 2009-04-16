package net.electroland.laface.tracking;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTrackerServer;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.laface.core.Impulse;
import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.shows.WaveShow;

/**
 * Measures track locations and creates a Mover object with location, velocity,
 * and acceleration / deceleration values to be used as an interpretive layer between
 * blob tracking and sprite drawing.
 * 
 * @author Aaron Siegel
 */

public class Tracker extends Thread implements TrackListener, MoverListener {
	
	private LAFACEMain main;
	private int sampleSize;
	private ConcurrentHashMap<Integer,Mover> movers;
	private ConcurrentHashMap<Integer,Candidate> candidates;
	private LinkedBlockingQueue<TrackResults> resultsQueue;
	private BlobTrackerServer bts;
	private boolean running = true;
	
	public Tracker(LAFACEMain main, int sampleSize){
		this.main = main;
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
						candidates.put(newtrack.id, new Candidate(newtrack.id));
						//System.out.println("new candidate! "+newtrack.id);
					}
				} else if(result.existing.size() > 0){				// EXISTING
					Iterator<Track> iter = result.existing.iterator();
					while(iter.hasNext()){
						Track track = iter.next();
						if(candidates.containsKey(track.id)){
							Candidate candidate = candidates.get(track.id);		// get the candidate that matches this track
							int srcWidth = Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
							int srcHeight = Integer.parseInt(ElProps.THE_PROPS.get("srcHeight").toString());
							candidate.addLocation(track.x/srcWidth, track.y/srcHeight);			// add the new location to history...
							if(candidate.getLocations().size() >= sampleSize){	// if it meets the sample size...
								if(!candidate.isStatic()){
									Mover m = new Mover(candidate);
									m.addListener(this);							// needed to remove from CHM
									movers.put(track.id, m);						// make a mover from this candidate
									candidates.remove(track.id);					// remove candidate
									if(main.getCurrentAnimation() instanceof WaveShow){
										Impulse impulse;
										if(track.x < Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString())/2){
											impulse = new Impulse(main, 0, 2000, false);
										} else {
											impulse = new Impulse(main, 0, 2000, true);
										}
										impulse.start();
									}
									//System.out.println("new MOVER! "+track.id);
								}
							}
						}
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
	}
	
	public void updateTracks(TrackResults results) {
		resultsQueue.offer(results);
	}

	public void moverEvent(Mover mover) {
		if(mover.isDead()){
			movers.remove(mover.getID());
		}
	}
	
	public ConcurrentHashMap<Integer,Mover> getMovers(){
		return movers;
	}

}
