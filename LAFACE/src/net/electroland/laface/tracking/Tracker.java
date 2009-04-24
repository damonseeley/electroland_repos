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

public class Tracker extends Thread implements TrackListener, MoverListener, TargetListener {
	
	private LAFACEMain main;
	private int sampleSize;
	private ConcurrentHashMap<Integer,Mover> movers;
	private ConcurrentHashMap<Integer,Target> targets;
	private ConcurrentHashMap<Integer,Candidate> candidates;
	private LinkedBlockingQueue<TrackResults> resultsQueue;
	private BlobTrackerServer bts;
	private long idleImpulseStart;
	private int idleImpulsePeriod;
	private boolean running = true;
	
	public Tracker(LAFACEMain main, int sampleSize){
		this.main = main;
		this.sampleSize = sampleSize;								// minimum number of location/time samples to create a mover
		movers = new ConcurrentHashMap<Integer,Mover>();
		targets = new ConcurrentHashMap<Integer,Target>();
		candidates = new ConcurrentHashMap<Integer,Candidate>();
		resultsQueue = new LinkedBlockingQueue<TrackResults>();		// used to get info on active tracks
		ElProps.init("depends//blobTracker.props");					// load tracking properties
		bts = new BlobTrackerServer(ElProps.THE_PROPS);				// launch the blob tracker
		bts.addTrackListener(0, this);
		idleImpulseStart = System.currentTimeMillis();
		idleImpulsePeriod = 10000;
	}
	
	public void run(){
		while(running){
			try {
				TrackResults result = resultsQueue.take();			// will block until something is on the queue
				
				if(result.created.size() > 0){						// CREATED
					Iterator<Track> iter = result.created.iterator();
					while(iter.hasNext()){
						Track track = iter.next();
						Target t = new Target(track); 				// create a new target associated with this track
						t.addListener(this);
						targets.put(track.id, t);
						
						// pulse for new intro
						if(main.getCurrentAnimation() instanceof WaveShow){
							Impulse impulse;
							if(track.x < Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString())/2){
								impulse = new Impulse(main, 0, 1000, false);
							} else {
								impulse = new Impulse(main, 0, 1000, true);
							}
							impulse.start();
						}
					}
				} else if(result.existing.size() > 0){				// EXISTING
					// targets update their own position
				} else if(result.deleted.size() > 0){				// DELETED
					Iterator<Track> iter = result.deleted.iterator();
					while(iter.hasNext()){
						Track track = iter.next();
						
						// TODO not getting deaths from every track, meaning targets never
						// switch to their automated vectors, never move beyond the raster
						// constraints, and never get destroyed.
						
						if(targets.containsKey(track.id)){
							Target t = targets.get(track.id);
							t.trackDied();
						}
					}
				}
				
				
				/** OLD CODE **/
				
				/*
				if(result.created.size() > 0){						// CREATED
					Iterator<Track> iter = result.created.iterator();
					while(iter.hasNext()){
						Track newtrack = iter.next();
						candidates.put(newtrack.id, new Candidate(newtrack));
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
							if(candidate.getLocations().size() >= sampleSize){		// if it meets the sample size...
								if(!candidate.isStatic() && candidate.endZone()){	// if it's moving at the minimum speed and in the appropriate zone...
									Mover m = new Mover(candidate);
									m.addListener(this);							// needed to remove from CHM
									movers.put(track.id, m);						// make a mover from this candidate
									candidates.remove(track.id);					// remove candidate
									if(main.getCurrentAnimation() instanceof WaveShow){
										Impulse impulse;
										if(track.x < Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString())/2){
											impulse = new Impulse(main, 0, 1000, false);
										} else {
											impulse = new Impulse(main, 0, 1000, true);
										}
										impulse.start();
									}
									//System.out.println("new MOVER! "+track.id);
								}
							}
						}
					}
				} else if(result.deleted.size() > 0){				// DELETED
					Iterator<Track> iter = result.deleted.iterator();
					while(iter.hasNext()){
						Track track = iter.next();
						if(movers.containsKey(track.id)){
							Mover m = movers.get(track.id);
							m.trackDied();
						}
					}
				} else if(result.existing.size() == 0){	// no cars around, so keep the wave active
					if(System.currentTimeMillis() - idleImpulseStart >= idleImpulsePeriod){		// every impulse period
						boolean direction = false;
						if(Math.random() > 0.5){
							direction = true;
						}
						Impulse impulse = new Impulse(main, 0, 2000, direction, 80, 70);			// create a mild wave
						impulse.start();
						idleImpulseStart = System.currentTimeMillis();
					}
				}
				*/
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
	
	public ConcurrentHashMap<Integer,Target> getTargets(){
		return targets;
	}
	
	public void addTrackListener(TrackListener listener){
		bts.addTrackListener(0, listener);
	}

	public void targetEvent(Target target) {
		//System.out.println("target "+target.getID() + " event");
		if(target.isDead()){
			targets.remove(target.getID());
			//System.out.println("target "+target.getID() + " removed");
		}
	}

}
