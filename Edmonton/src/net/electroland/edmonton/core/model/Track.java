package net.electroland.edmonton.core.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.log4j.Logger;


public class Track {

	private ArrayList<TrackEvent> trackEvents;
	public long startTime;
	public long lastTrackUpdate,lastUpdate;
	public double x;
	public double xSpeed;	// in world units per second
	public double speedAdjust;
	public double sDistRev,sDistFwd; 	// distance to search for new trackEvents
	public long sTime; 	// timeout for tracks to search for new events

	private static Logger logger = Logger.getLogger(Track.class);


	public Track(double x) {
		trackEvents = new ArrayList<TrackEvent>();
		startTime = System.currentTimeMillis();
		lastTrackUpdate = startTime;
		lastUpdate = startTime;
		trackEvents.add(new TrackEvent(startTime,x));
		sDistRev = 7.5; // slightly more than the 7.39 dist from sensor to sensor
		sDistFwd = 2.0; //seach forward to find interpolation overshoot tracks
		sTime = 2000; 	// (ms) start with 2 seconds
		xSpeed = 5.8; // unit per second, based on 4.53 feet per second, in Meters * world unit multiplier (4 as of this writing)
		speedAdjust = 1.0;
	}

	public void update() {
		// for now, simply update based on init speed
		// eventually calculate based on the last n events
			long tDelta = System.currentTimeMillis() - lastUpdate; // time since last update
			double xDelta = xSpeed*tDelta/1000; //distance to travel
			x -= xDelta;
			//logger.info("Tracker: updated track " + this + " to x value " + x + " with delta " + xDelta);
			lastUpdate = System.currentTimeMillis();
		
	}

	public void newTrackEvent(double x) {
		this.x = x;
		lastTrackUpdate = System.currentTimeMillis();
		trackEvents.add(new TrackEvent(lastTrackUpdate,x));

		// update speed variable if more than three trackEvents
		if (trackEvents.size() > 2){
			Long totalTime = trackEvents.get(trackEvents.size()-1).timestamp - trackEvents.get(0).timestamp;
			double totalDist = trackEvents.get(0).x - trackEvents.get(trackEvents.size()-1).x;
			double newSpeed = totalDist/(totalTime/1000);
			if (newSpeed > 0.5){
				xSpeed = newSpeed * speedAdjust;
			}
			//xSpeed = totalDist/(totalTime/1000); //divide total distance from events by total time/1000 to get xSpeed as unit per second 
			logger.info("Tracker: updated xSpeed " + xSpeed);

		}
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





