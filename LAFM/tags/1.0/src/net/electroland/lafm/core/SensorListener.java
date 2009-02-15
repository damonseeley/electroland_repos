package net.electroland.lafm.core;

import net.electroland.detector.DMXLightingFixture;

public interface SensorListener {
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn);
}