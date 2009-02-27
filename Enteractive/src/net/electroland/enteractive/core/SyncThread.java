package net.electroland.enteractive.core;

import java.util.Iterator;
import java.util.List;
import net.electroland.detector.DMXLightingFixture;

/**
 * Used to sync fixtures with current show/transition, regulates frame rate
 */

public class SyncThread extends Thread{
	
	Show show;							// show or transition/blend object
	List<DMXLightingFixture> fixtures;	// floor, facade, and screen fixtures
	
	public SyncThread(List<DMXLightingFixture> fixtures){
		this.fixtures = fixtures;
	}

	public void run(){
		while(true){
			//Raster raster = show.getRasterFrame(Model m);
			Raster raster = show.getRasterFrame();	// place holder
			Iterator <DMXLightingFixture> i = fixtures.iterator();
			while (i.hasNext()){
				i.next().sync(raster.getRaster());
			}
			try {
				// TODO: Add the delay compensation here to retain frame rate
				sleep(33);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
}
