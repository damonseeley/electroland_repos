package net.electroland.enteractive.core;

import java.util.Map;

/**
 * Handles any adjustments to the model based on sensor activity sent from UDPParser.
 * @author asiegel
 */

public class PersonTracker {
	
	private Model model;

	public PersonTracker(int gridWidth, int gridHeight){
		model = new Model(gridWidth, gridHeight);
	}
	
	public Model getModel(){
		return model;
	}
	
	public void updateSensors(int offset, byte[] data, Map<Integer, Tile> stuckTiles){
		boolean[] newdata = new boolean[data.length];
		for(int i=0; i<data.length; i++){
			if((int)(data[i] & 0xFF) > 0){
				newdata[i] = true;
			} else {
				newdata[i] = false;
			}
		}
		model.updateSensors(offset, newdata, stuckTiles);
	}
	
	public void updateAverage(double average){
		model.updateAverage(average);
	}
	
}
