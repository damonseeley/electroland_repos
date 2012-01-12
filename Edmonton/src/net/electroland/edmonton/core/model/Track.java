package net.electroland.edmonton.core.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.log4j.Logger;


public class Track {

	public int id;
	private ArrayList<TrackEvent> trackEvents;
	public long startTime;
	public long lastTrackUpdate,lastUpdate;
	public String lastIState;
	public double x;
	public double xSpeed;	// in world units per second
	public double speedAdjust;
	public double fwdSearchDist; // distance in front of a track (and behind a sensor) to search
	public double revSearchDist; // distance behind a track (and forward of a sensor) to search (less common)
	private double fwdSearchOrig, revSearchOrig;
	private double searchDistInc;
	public long sTime; 	// timeout before tracks are deleted
	public float staleness;

	private static Logger logger = Logger.getLogger(Track.class);


	public Track(int id, double x, String isID) {
		this.id = id;
		this.x = x;
		this.lastIState = isID;
		trackEvents = new ArrayList<TrackEvent>();
		startTime = System.currentTimeMillis();
		lastTrackUpdate = startTime;
		lastUpdate = startTime;
		trackEvents.add(new TrackEvent(startTime,x));
		fwdSearchDist = 7.5; // slightly more than the 7.39 dist from sensor to sensor
		fwdSearchDist = 7.0; // slightly more than the 7.39 dist from sensor to sensor
		revSearchDist = -3.0; //seach forward to find interpolation overshoot tracks
		revSearchDist = -7.0; //seach forward to find interpolation overshoot tracks
		fwdSearchOrig = fwdSearchDist;
		revSearchOrig = revSearchDist;
		searchDistInc = 0.1;
		sTime = 3500; 	// (ms) start with 2 seconds
		xSpeed = 5.5; // unit per second, based on 4.53 feet per second, in Meters * world unit multiplier (4 as of this writing)
		speedAdjust = 1.0;
		staleness = 1.0f; // where 1.0 means not stale at all
	}

	public void update() {

		long tDelta = System.currentTimeMillis() - lastUpdate; // time since last update
		long tUpdateDelta = System.currentTimeMillis() - lastTrackUpdate;
		staleness = 1.0f - (float)tUpdateDelta/(float)sTime;
		//logger.info(tUpdateDelta + " " + sTime + " " + staleness);
		
		// grow the search distance by the increment value if no matching has occurred (well, by default)
		fwdSearchDist += searchDistInc;
		revSearchDist -= searchDistInc;

		// update speed variable if more than three trackEvents
		if (trackEvents.size() > 2){
			Long totalTime = trackEvents.get(trackEvents.size()-1).timestamp - trackEvents.get(0).timestamp;
			double totalDist = trackEvents.get(0).x - trackEvents.get(trackEvents.size()-1).x;
			double newSpeed = totalDist/(totalTime/1000);
			if (newSpeed > 0.5){
				xSpeed = newSpeed * speedAdjust;
			}
			//logger.info("Tracker: updated xSpeed " + xSpeed);
		}

		double xDelta = xSpeed*tDelta/1000; //distance to travel
		x -= xDelta;
		//logger.info("Tracker: updated track " + this + " to x value " + x + " with delta " + xDelta);
		lastUpdate = System.currentTimeMillis();

	}

	public void newTrackEvent(double x) {
		lastTrackUpdate = System.currentTimeMillis();
		trackEvents.add(new TrackEvent(lastTrackUpdate,x));
	}

	public void newTrackEvent(double nx, String isID) {

		if (isID.equals(lastIState)) {
			//logger.debug("Same IState ID reported in Track");
			lastIState = isID;
		} else {
			//logger.debug("New IState reported in Track with ID " + isID);
			newTrackEvent(nx);
			this.x = nx;
			lastIState = isID;
		}
		
		//reset search distances
		fwdSearchDist = fwdSearchOrig;
		revSearchDist = revSearchOrig;
		
		//newTrackEvent(nx);

		
	}

	public boolean isExpired() {
		if ((System.currentTimeMillis() - lastTrackUpdate) > sTime){
			return true;
		} else {
			return false;
		}
	}


}

class TrackEvent
{
	Long timestamp;
	double x;

	public TrackEvent(Long t, double x){
		this.timestamp = t;
		this.x = x;
	}
}





