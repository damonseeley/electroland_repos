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



public class TrackerModelWatcher extends ModelWatcher {
	
	private static Logger logger = Logger.getLogger(TrackerModelWatcher.class);

	private List<Track> tracks;
	private Hashtable<String, Object> context;

	/**
	 * 
	 */
	public TrackerModelWatcher(Hashtable<String, Object> context)
	{
		//tracks = new ArrayList<Track>();
		tracks = Collections.synchronizedList(new ArrayList<Track>());
		this.context = context;
		context.put("tracks", tracks);
	}

	/**
	 *
	 */
	@Override
	public boolean poll() {

		// iterate through states
		for (IState state : this.getStates())
		{
			if (state.getState()){
				// if a state is on do a search for nearby tracks, if tracks is not empty
				if (tracks.size() > 0){
					// search for nearby existing track
					for (Track t : tracks)
					{
						if ((t.x - state.getLocation().x) < t.sDistRev || (state.getLocation().x - t.x) < t.sDistFwd) {
							// if a track is found within search update the track
							t.newTrackEvent(state.getLocation().x);
							break;
						}
					}
				} else {
					// if not then create a new track
					tracks.add(new Track(state.getLocation().x));
				}
			}
		}


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
						if (tr.x < 0){
							tracks.remove(tr);
						}
						//remove them if their searchtime has been exceeded
						if (tr.isExpired()){
							tracks.remove(tr);
						}
					}

				}
			}
		} catch (ConcurrentModificationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 

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

