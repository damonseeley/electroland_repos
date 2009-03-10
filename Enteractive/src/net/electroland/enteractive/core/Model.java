package net.electroland.enteractive.core;

public class Model {
	
	private boolean[] sensors;
	
	public Model(int gridWidth, int gridHeight){
		sensors = new boolean[gridWidth*gridHeight];
		for(int i=0; i<sensors.length; i++){
			sensors[i] = false;
		}
	}
	
	public void updateSensors(int offset, boolean[] data){
		System.arraycopy(data, 0, sensors, offset, data.length);
	}
	
	public boolean[] getSensors(){
		return sensors;
	}

}
