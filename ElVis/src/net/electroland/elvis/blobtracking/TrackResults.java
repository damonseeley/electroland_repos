package net.electroland.elvis.blobtracking;

import java.util.StringTokenizer;
import java.util.Vector;

import net.electroland.elvis.net.StringAppender;

public class TrackResults<T extends BaseTrack> implements StringAppender {
	public Vector<T> created;
	public Vector<T> existing;
	public Vector<T> deleted;

	public  TrackResults(Vector<T> c, Vector<T> e, Vector<T> d) {
		created = c;
		existing = e;
		deleted = d;
	}

	public TrackResults() {
		this(new Vector<T>(), new Vector<T>(), new Vector<T>());
	}

	@Override
	public void buildString(StringBuilder sb) {
		for(BaseTrack t : created) { 
			t.buildString(sb);
			sb.append(",");
		}
		sb.append("|");
		for(BaseTrack t : existing) { 
			sb.append(",");
			t.buildString(sb);
		}
		sb.append(",");
		sb.append("|");
		for(BaseTrack t : deleted) { 
			sb.append(",");
			t.buildString(sb);
		}

	}
	public String toString() {
		StringBuilder sb = new StringBuilder();
		buildString(sb);
		return sb.toString();
		
	}
	public static TrackResults<BaseTrack> buildFromString(StringTokenizer tokenizer) {
		Vector<BaseTrack> created = new Vector<BaseTrack>();
		Vector<BaseTrack> existing = new Vector<BaseTrack>();
		Vector<BaseTrack> deleted = new Vector<BaseTrack>();		

		
		BaseTrack t = BaseTrack.buildFromTokenizer(tokenizer);
		while(t != null) {
			created.add(t);
			t = BaseTrack.buildFromTokenizer(tokenizer);
		}
		t = BaseTrack.buildFromTokenizer(tokenizer);
		while(t != null) {
			existing.add(t);
			t = BaseTrack.buildFromTokenizer(tokenizer);
		}
		t = BaseTrack.buildFromTokenizer(tokenizer);
		while(t != null) {
			deleted.add(t);
			t = BaseTrack.buildFromTokenizer(tokenizer);
		}
		return new TrackResults<BaseTrack>(created, existing, deleted);		
		
	}

}
