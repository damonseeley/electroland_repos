package net.electroland.elvis.blobtracking;

import java.util.Vector;

public class TrackResults {
	public Vector<Track> created;
	public Vector<Track> existing;
	public Vector<Track> deleted;
	
	public  TrackResults(Vector<Track> c, Vector<Track> e, Vector<Track> d) {
		created = c;
		existing = e;
		deleted = d;
	}

}
