package net.electroland.enteractive.core;

public interface SensorListener {
	abstract void sensorEvent();	// must pass Sensor object with x/y, on/off
}
