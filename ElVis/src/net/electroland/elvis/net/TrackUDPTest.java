package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Vector;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;

// quick test to make sure tracks broadcaster and client work (at least on localhost)
public class TrackUDPTest {
	public static final int TEST_PORT = 3434;
	TrackUDPBroadcaster broadcaster;
	TrackUDPClient client;
	
	
	public TrackUDPTest() throws SocketException, UnknownHostException {
		broadcaster = new TrackUDPBroadcaster(TEST_PORT);
		client = new TrackUDPClient(TEST_PORT) {
			public void handle(TrackResults<BaseTrack> t) {
				System.out.println(t);
			}

		};
	}
	public void test() {
		broadcaster.start();
		client.start();
		
		for(int i = 0; i < 20; i++) {
			Vector<BaseTrack> c = new Vector<BaseTrack>();
			Vector<BaseTrack> e = new Vector<BaseTrack>();
			Vector<BaseTrack> d = new Vector<BaseTrack>();
			
			for(int j = 0; j < 10; j++) {
				BaseTrack t = new BaseTrack(j, (float)i, (float)j+2, j %2 == 0);		
				c.add(t);
			}
			for(int j = 0; j < 10; j++) {
				BaseTrack t = new BaseTrack(j, (float)i, (float)j+2, j %2 == 0);		
				e.add(t);
			}
			for(int j = 0; j < 10; j++) {
				BaseTrack t = new BaseTrack(j, (float)j+1, (float)j+2, j %2 == 0);		
				d.add(t);
			}
			TrackResults tr = new TrackResults(c,e,d);
			broadcaster.updateTracks(tr);
		}
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		broadcaster.stopRunning();
		client.stopRunning();
	}
	
	public static void main(String[] arg) throws SocketException, UnknownHostException {
		new TrackUDPTest().test();
	}

}
