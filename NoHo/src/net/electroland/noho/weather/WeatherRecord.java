package net.electroland.noho.weather;

import java.util.Calendar;

public class WeatherRecord {
	Calendar sunrise;
	Calendar sunset;
	float visibility;
	float outsidetemp;
	float artboxtemp;
	int condition;
	
	public int getCondition() {
		return condition;
	}
	public Calendar getSunrise() {
		return sunrise;
	}
	public Calendar getSunset() {
		return sunset;
	}
	public float getOutsideTemperature() {
		return outsidetemp;
	}
	public float getArtboxTemperature() {
		return artboxtemp;
	}
	public float getVisibility() {
		return visibility;
	}
}
