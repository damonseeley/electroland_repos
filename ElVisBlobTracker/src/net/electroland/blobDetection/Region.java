package net.electroland.blobDetection;

import net.electroland.blobTracker.util.ElProps;

public class Region {

	public int id;
	public int minBlobSize;
	public int maxBlobSize;
	public float maxTrackMove;
	public float nonMatchPenalty;
	//public float provisionalPentaly;
	public float provisionalPenalty;
	
	public int framesUntilCertainTrack;
	public int framesUntilDeleteTrack; // persistance
	
	public int maxClusteringSize;
	public float maxClusteringDist;
	
	public float velocityMatchPercentage;

	
	public Region(int i) {
		id = i;
		minBlobSize = ElProps.THE_PROPS.getProperty("minBlobSize" + i, 50);
		maxBlobSize = ElProps.THE_PROPS.getProperty("maxBlobSize" + i, 400);
		maxTrackMove = ElProps.THE_PROPS.getProperty("maxTrackMove" + i, 25);
		nonMatchPenalty = ElProps.THE_PROPS.getProperty("nonMatchPenalty" + i, 2.0f * maxTrackMove);
		//provisionalPentaly = ElProps.THE_PROPS.getProperty("provisionalPentaly" + i, maxTrackMove);
		provisionalPenalty = ElProps.THE_PROPS.getProperty("provisionalPenalty" + i, maxTrackMove);
		framesUntilCertainTrack = ElProps.THE_PROPS.getProperty("framesUntilCertainTrack" + i, 20);
		framesUntilDeleteTrack = ElProps.THE_PROPS.getProperty("framesUntilDeleteTrack" + i, 40);
		
		
		maxClusteringSize = ElProps.THE_PROPS.getProperty("maxClusteringSize" + i, -1);
		maxClusteringDist = ElProps.THE_PROPS.getProperty("maxClusteringDist" + i, 0);
		velocityMatchPercentage = ElProps.THE_PROPS.getProperty("velocityMatchPercentage" + i, -1.0f);
		
	}

}
