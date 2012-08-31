package net.electroland.elvis.regions;

import java.util.StringTokenizer;
import java.util.Vector;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.net.StringAppender;

public class PolyRegionResults<T extends BasePolyReagion > implements StringAppender {
	Vector<T> regions;
	
	public PolyRegionResults() {
		this( new Vector<T>());
	}
	public PolyRegionResults(Vector<T> regions) {
		this.regions = regions;

	}
	@Override
	public void buildString(StringBuilder sb) {
		for(BasePolyReagion r : regions) {
			r.buildString(sb);
			sb.append(",");
		}		
	}

	public static String buildString(Vector<PolyRegion> regions) {
		StringBuilder sb = new StringBuilder();
		for(BasePolyReagion r : regions) {
			r.buildString(sb);
			sb.append(",");
		}		
		return sb.toString();		
	}

	public static PolyRegionResults<BasePolyReagion> buildFromStringTokenizer(StringTokenizer t) {
		Vector<BasePolyReagion> regions = new Vector<BasePolyReagion>();
		BasePolyReagion r = BasePolyReagion.buildFromTokenizer(t);
		while(r != null) {
			regions.add(r);
			r = BasePolyReagion.buildFromTokenizer(t);
		}
		return new PolyRegionResults(regions);

	}

}
