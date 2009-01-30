package net.electroland.lafm.shows;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.ShowThreadListener;
import net.electroland.lafm.core.SoundManager;

public class SuperShowThread extends ShowThread implements ShowThreadListener{
	
	private List<List<ShowThread>> showArrangement;		// 2D List of shows
	private List<DMXLightingFixture> availableFixtures;	// collects fixtures as sessions complete
	private int currentSession;							// keeps track of currently running session

	public SuperShowThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		showArrangement = Collections.synchronizedList(new ArrayList<List<ShowThread>>());
		currentSession = 0;
	}
	
	public void addShow(int session, ShowThread newShow){
		if(showArrangement.size() > session){				// if list already contains list...
			showArrangement.get(session).add(newShow);		// append show to that session
		} else {											// if session doesn't exist...
			showArrangement.add(session, Collections.synchronizedList(new ArrayList<ShowThread>()));	// create new session
		}
	}

	@Override
	public void complete(PGraphics raster) {
		// TODO Auto-generated method stub

	}

	@Override
	public void doWork(PGraphics raster) {
		if(availableFixtures.size() == 22){		// if all fixtures available
		
		}
	}

	@Override
	public void notifyComplete(ShowThread showthread, Collection<DMXLightingFixture> flowers) {
		// TODO If all fixtures are freed, launch the next session of shows
		// TODO If all sessions complete, exit the supershow
		Iterator<DMXLightingFixture> iter = flowers.iterator();
		while(iter.hasNext()){
			DMXLightingFixture flower = iter.next();
			if(!availableFixtures.contains(flower)){
				availableFixtures.add(flower);
			}
		}
		//availableFixtures.addAll(flowers);	// potentially unsafe?
		if(availableFixtures.size() == 22){		// if all fixtures recollected...
			currentSession++;					// go to next session...
			if(currentSession == showArrangement.size()){	// if session limit reached...
				cleanStop();					// exit the super show
			}
		}
	}

}
