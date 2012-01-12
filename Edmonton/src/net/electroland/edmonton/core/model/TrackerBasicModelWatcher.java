package net.electroland.edmonton.core.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

import org.apache.log4j.Logger;



public class TrackerBasicModelWatcher extends ModelWatcher {

	private static Logger logger = Logger.getLogger(TrackerBasicModelWatcher.class);

	private List<Track> tracks;
	private Hashtable<String, Object> context;
	private int trackID = 0;

	/**
	 * 
	 */
	public TrackerBasicModelWatcher(Hashtable<String, Object> context)
	{
		//tracks = new ArrayList<Track>();
		tracks = Collections.synchronizedList(new ArrayList<Track>());
		this.context = context;
		context.put("tracks", tracks);
	}

	public void newTrack(double nx, String sID){
		tracks.add(new Track(trackID,nx,sID));
		trackID++;
	}

	public List<Track> getAllTracks(){
		return tracks;
	}

	/**
	 *
	 */
	@Override
	public boolean poll() {

		// update all tracks
		try {
			for (Track t : tracks)
			{
				//  iterate through tracks
				if (tracks.size() > 0){
					for (Track tr : tracks)
					{
						// update tracks if they were not updated by the previous loop
						tr.update();
						// remove track if x is below some threshold
						if (tr.x < -10){
							logger.debug("track " + tr.id + " left the scene and will be removed");
							tracks.remove(tr);
						}
						//remove them if their searchtime has been exceeded
						if (tr.isExpired()){
							logger.debug("track " + tr.id + " expired unmatched and will be removed");
							tracks.remove(tr);
						}
					}

				}
			}
		} catch (ConcurrentModificationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 


		// iterate through states and try to match to tracks
		for (IState s : this.getStates())
		{
			if (s.getState()){
				// logger.debug("state " + state.getID() + " is on");
				// if a state is on do a search for nearby tracks, if tracks is not empty
				// search for nearby existing track
				boolean matchedTrack = false;
				for (Track t : tracks)
				{
					// check condition of being short of sensor but within fwdSearchDist
					double tDelta = t.x - s.getLocation().x; //signed distance from track x to sensor x
					if (tDelta < t.fwdSearchDist && tDelta > 0) {
						// if a track is found within search update the track
						//logger.debug("matched track " + t.id + " @x=" + t.x + " with sensor " + s.getID() + " @x=" + s.getLocation().x);
						t.newTrackEvent(s.getLocation().x,s.getID());
						matchedTrack = true;
						break;
					}
					// check condition of being just past the sensor but within revSearchDist
					if (tDelta > t.revSearchDist && tDelta < 0) {
						// if a track is found within search update the track
						//logger.debug("matched track " + t.id + " @x=" + t.x + " with sensor " + s.getID() + " @x=" + s.getLocation().x);
						t.newTrackEvent(s.getLocation().x,s.getID());
						matchedTrack = true;
						break;
					}
				}
				// no track was found in the prev loop so add one
				if (!matchedTrack){
					logger.debug("no track found for state " + s.getID() + " so creating new track at x=" + s.getLocation().x);
					newTrack(s.getLocation().x,s.getID());
				}
			}
		}


		//logger.debug("tracks size: " + tracks.size());

		return true;
	}

	/**
	 * In the event, retuns a list of tracks, stored as a single elements in a map per 
	 * this methods super requirements 
	 */
	@Override
	public Map<String, Object> getOptionalPositiveDetails() {
		Map<String,Object> tmap = new HashMap<String,Object>();
		tmap.put("Tracks", tracks);
		return tmap;
	}
}

