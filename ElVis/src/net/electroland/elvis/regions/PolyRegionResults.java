package net.electroland.elvis.regions;

import java.util.StringTokenizer;
import java.util.Vector;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.net.StringAppender;

public class PolyRegionResults<T extends BasePolyRegion > implements StringAppender {
	Vector<T> regions;
	
	public PolyRegionResults() {
		this( new Vector<T>());
	}
	public PolyRegionResults(Vector<T> regions) {
		this.regions = regions;

	}
	@Override
	public void buildString(StringBuilder sb) {
		for(BasePolyRegion r : regions) {
			r.buildString(sb);
			sb.append(",");
		}		
	}

	public static String buildString(Vector<PolyRegion> regions) {
		StringBuilder sb = new StringBuilder();
		for(BasePolyRegion r : regions) {
			r.buildString(sb);
			sb.append(",");
		}		
		return sb.toString();		
	}

	public static PolyRegionResults<BasePolyRegion> buildFromStringTokenizer(StringTokenizer t) {
		Vector<BasePolyRegion> regions = new Vector<BasePolyRegion>();
		BasePolyRegion r = BasePolyRegion.buildFromTokenizer(t);
		while(r != null) {
			regions.add(r);
			r = BasePolyRegion.buildFromTokenizer(t);
		}
		return new PolyRegionResults(regions);

	}

}
