package net.electroland.lafm.weather;

public interface WeatherChangeListener {
	
	
	/**
	 * should return relatively quickly
	 * @param wce
	 */
	public void weatherChanged(WeatherChangedEvent wce);
	
	public void tempUpdate(float tu);
	
}
