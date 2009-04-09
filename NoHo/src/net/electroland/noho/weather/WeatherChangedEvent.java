package net.electroland.noho.weather;

import java.util.Calendar;


public class WeatherChangedEvent {
	
	boolean hasSunriseChanged = false;
	boolean hasSunsetChanged = false;
	boolean hasConditionChanged = false;
	boolean hasVisibilityChanged = false;
	boolean hasTemperatureChanged = false;

	protected WeatherRecord record;
	
	/**
	 * 
	 * @param record
	 * @param newRecord
	 * @return null if no diff between record and newRecord else returns WetherChangedEvent
	 */
	public static WeatherChangedEvent generate(WeatherRecord record, WeatherRecord newRecord) { // returns true if there is a change
		WeatherChangedEvent wce = new WeatherChangedEvent();
		
		if(record == null) {
			wce.hasSunriseChanged = true;
			wce.hasSunsetChanged = true;
			wce.hasConditionChanged = true;
			wce.hasVisibilityChanged = true;
			wce.hasTemperatureChanged = true;
			wce.record = newRecord;
			return wce;
		} 
		
		wce.hasSunriseChanged = record.sunrise.get(Calendar.HOUR_OF_DAY) != newRecord.sunrise.get(Calendar.HOUR_OF_DAY);
		wce.hasSunriseChanged = wce.hasSunriseChanged || (record.sunrise.get(Calendar.MINUTE) != newRecord.sunrise.get(Calendar.MINUTE));
		// don't bother checking am/pm
		wce.hasSunsetChanged = record.sunset.get(Calendar.HOUR_OF_DAY) != newRecord.sunset.get(Calendar.HOUR_OF_DAY);
		wce.hasSunsetChanged = wce.hasSunsetChanged || (record.sunset.get(Calendar.MINUTE) != newRecord.sunset.get(Calendar.MINUTE));
		
		wce.hasConditionChanged = record.condition != newRecord.condition;
		
		wce.hasVisibilityChanged = record.visibility != newRecord.visibility;
		
		wce.hasTemperatureChanged = ( record.outsidetemp != newRecord.outsidetemp);
		if(wce.hasChange()) {
			wce.record = newRecord;
			return wce;
		} else {
			return null;
		}
		
	}
	
	
	public WeatherRecord getWeatherRecord() {
		return record;
	}
	
	public boolean hasChange() {
		return hasSunriseChanged || hasSunsetChanged || hasConditionChanged ||
				hasVisibilityChanged || hasTemperatureChanged;
	}


	public boolean hasConditionChanged() {
		return hasConditionChanged;
	}


	public boolean hasSunriseChanged() {
		return hasSunriseChanged;
	}


	public boolean hasSunsetChanged() {
		return hasSunsetChanged;
	}


	public boolean hasTemperatureChanged() {
		return hasTemperatureChanged;
	}


	public boolean hasVisibilityChanged() {
		return hasVisibilityChanged;
	}


	public WeatherRecord getRecord() {
		return record;
	}
	
}
