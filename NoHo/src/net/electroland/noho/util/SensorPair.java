package net.electroland.noho.util;

public class SensorPair {
	
	public long startTime = -1;
	public boolean waiting = false;

	public int startSensorId;
	public int endSensorId;
	public int type = -1;
	public int id = -1;
	public double tmultiplier;
	public double threshold;
	
	public SensorPair(int startSensorId, int endSensorId, double threshold,
						int type, int id, double tmultiplier){
		this.startSensorId = startSensorId;
		this.endSensorId = endSensorId;
		this.type = type;
		this.threshold = threshold;
		this.id = id;
		this.tmultiplier = tmultiplier;
	}
}