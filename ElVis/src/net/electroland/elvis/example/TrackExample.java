package net.electroland.elvis.example;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.elvis.blobktracking.core.ElVisServer;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.blobtracking.TrackResults;
import net.electroland.elvis.util.ElProps;

public class TrackExample extends Thread implements TrackListener {
	LinkedBlockingQueue<TrackResults> resultsQueue = new LinkedBlockingQueue<TrackResults>();

	/**
	 * @param args
	 * @throws UnknownHostException 
	 * @throws SocketException 
	 */
	public static void main(String[] args) throws SocketException, UnknownHostException {
		ElProps p;
		if(args.length > 0) {
			p = ElProps.init(args[0]);
		} else {
			p = ElProps.init("depends/blobTracker.props");
		}

		ElVisServer bts = new ElVisServer(
				p
		);
		TrackExample example = new TrackExample();
		//adding to region 1, not 0
		bts.addTrackListener(example);
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
