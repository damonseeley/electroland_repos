package net.electroland.elvis.blobtracking;

import java.util.Map;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.elvis.blobtracking.CSP.Solution;
import net.electroland.elvis.util.ElProps;

public class Tracker extends Thread {

//	float maxClusteringSize = -1;
	boolean isRunning = true;
	public int framesUntilCertainTrack;
	public int framesUntilDeleteTrack; // persistance
	public float velocityMatchPercentage;


	
	ElProps props;
	public Vector<Track> tracks = new Vector<Track>();

	Vector<TrackListener> trackListeners = new Vector<TrackListener>();


	LinkedBlockingQueue<Vector<Blob>> blobsQueue = new LinkedBlockingQueue<Vector<Blob>>();

	public CSP csp;
	public Grid grid;
	Grid clusterGrid;


	public Tracker(ElProps props) {
		this.props = props;
		
//		framesUntilCertainTrack = ElProps.THE_PROPS.getProperty("framesUntilCertainTrack" , 20);
		framesUntilCertainTrack = props.getProperty("framesUntilCertainTrack" , 20);
		framesUntilDeleteTrack = props.getProperty("framesUntilDeleteTrack" , 40);
		velocityMatchPercentage = props.getProperty("velocityMatchPercentage", -1.0f);
		
		csp = new CSP();
		float maxTrackMove =props.getProperty("maxTrackMove", 50);
		float nonMatchPenalty = props.getProperty("nonMatchPenalty", 2.0f * maxTrackMove);
		float provisionalPenalty = props.getProperty("provisionalPenalty", maxTrackMove);

		
		csp.setNonMatchPenalty(nonMatchPenalty);
		csp.setProvisionalPenalty(provisionalPenalty);
		/*
		int maxClusteringSize = props.getProperty("maxClusteringSize", -1);
		if(maxClusteringSize > 0) {

			clusterGrid = new Grid(
					props.getProperty("srcWidth", 320),
					props.getProperty("srcWidth", 320),
					props.getProperty("maxClusteringDist", 0));
			clusterGrid.setClusterSize(maxClusteringSize);
		}
		*/

//		maxTrackMove0
		grid = new Grid(
				props.getProperty("srcWidth", 320),
				props.getProperty("srcWidth", 240),
				maxTrackMove);

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
			Track t = new Track(framesUntilCertainTrack, framesUntilDeleteTrack, velocityMatchPercentage);
			t.setBlobLoc(b);
			existingTracks.add(t);
			createdTracks.add(t);
		}
		tracks = existingTracks;

		for(TrackListener l : trackListeners) {
//			System.out.println("newtracks" + newTracks.size());
			l.updateTracks(new TrackResults<Track>(createdTracks, existingTracks, deletedTracks));
		}

	}
	public void run() {
		while(isRunning) {
			try {
				Vector<Blob> blobs = blobsQueue.take();
				/*
				if(region.maxClusteringSize > 0) {
					clusterGrid.clear();
					clusterGrid.addBlobs(blobs);
					grid.clear();
					grid.addBlobs(clusterGrid.mergedBlobs());
				} else {
					*/
					grid.clear();
					grid.addBlobs(blobs);					
				


				updateTracks(csp.solve(grid, tracks, ! blobsQueue.isEmpty())); 
				// if the queue not empty perform a single pass 
				// else search for a good solution until the next frame comes in


			} catch (InterruptedException e) {
			}
		}

	}
}
