package net.electroland.blobDetection.example;

import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.blobDetection.Blob;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTrackerServer;
import net.electroland.blobTracker.util.ElProps;

public class TrackExample extends Thread implements TrackListener {
	LinkedBlockingQueue<TrackResults> resultsQueue = new LinkedBlockingQueue<TrackResults>();

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		if(args.length > 0) {
			ElProps.init(args[0]);
		} else {
			ElProps.init("blobTracker.props");
		}

		BlobTrackerServer bts = new BlobTrackerServer(
				ElProps.THE_PROPS
		);
		TrackExample example = new TrackExample();
		bts.addTrackListener(1, example);
		example.start();
	}
	
	public void run() {
		while(true) {
			try {
				TrackResults result = resultsQueue.take(); // will block until someting is on the que
				//System.out.println(result.created.size() + " tracks created");
				if (result.created.size() > 0 || result.deleted.size() > 0) {
					System.out.println(result.existing.size() + " tracks");
				}
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public void updateTracks(TrackResults results) {
		resultsQueue.offer(results);		
	}

}
