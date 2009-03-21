package net.electroland.blobDetection.match;

import java.util.Map;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.Blob;
import net.electroland.blobDetection.Region;
import net.electroland.blobDetection.match.CSP.Solution;
import net.electroland.blobTracker.util.ElProps;

public class Tracker extends Thread {

	boolean isRunning = true;

	ElProps props;
	Vector<Track> tracks = new Vector<Track>();

	Vector<TrackListener> trackListeners = new Vector<TrackListener>();


	LinkedBlockingQueue<Vector<Blob>> blobsQueue = new LinkedBlockingQueue<Vector<Blob>>();

	CSP csp;
	Grid grid;

	Region region;

	public Tracker(ElProps props, Region region) {
		this.props = props;
		this.region = region;
		
		csp = new CSP(region);
		grid = new Grid(
				props.getProperty("srcWidth", 320),
				props.getProperty("srcWidth", 240),
				region.maxTrackMove);
	}

	public void addListener(TrackListener l) {
		trackListeners.add(l);
	}

	public void removeListener(TrackListener l) {
		trackListeners.remove(l);
	}


	public void queueBlobs(Vector<Blob> blobs) {
		csp.stopProcessing(); 
		// if searching an optimal solution stop and return best so far
		blobsQueue.add(blobs);
	}

	public void updateTracks(Solution solution) {
		
		Vector<Track> existingTracks = new Vector<Track>(tracks.size() + solution.unmachedBlobs.size());
		Vector<Track> deletedTracks = new Vector<Track>();
		Vector<Track> createdTracks = new Vector<Track>();
		
		for(Map.Entry<Track, Blob> assignment : solution.assigned.entrySet()) {
			Track t=assignment.getKey();
			t.setBlobLoc(assignment.getValue());
			if(! t.isRemoved) {
				existingTracks.add(t);				
			}

		}
		for(Track t : solution.tracks.keySet()) { // these are unmatched tracks 
			t.setBlobLoc(CSP.UNMATCHED);
			if(! t.isRemoved) {
				existingTracks.add(t);				
			} else {
				deletedTracks.add(t);
			}
		}
		for(Blob b : solution.unmachedBlobs) {
			Track t = new Track(region.framesUntilCertainTrack, region.framesUntilDeleteTrack);
			t.setBlobLoc(b);
			existingTracks.add(t);
			createdTracks.add(t);
		}
		tracks = existingTracks;

		for(TrackListener l : trackListeners) {
//			System.out.println("newtracks" + newTracks.size());
			l.updateTracks(new TrackResults(createdTracks, existingTracks, deletedTracks));
		}

	}
	public void run() {
		while(isRunning) {
			try {
				Vector<Blob> blobs = blobsQueue.take();
				grid.clear();
				grid.addBlobs(blobs);
				updateTracks(csp.solve(grid, tracks, ! blobs.isEmpty())); 
				// if the queue not empty perform a single pass 
				// else search for a good solution until the next frame comes in


			} catch (InterruptedException e) {
			}
		}

	}
}
