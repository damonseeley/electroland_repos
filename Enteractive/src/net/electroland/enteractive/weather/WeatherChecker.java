package net.electroland.enteractive.weather;

import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

import net.electroland.enteractive.core.EnteractiveMain;

import org.apache.log4j.Logger;

public class WeatherChecker  {
	protected static final Timer TIMER = new Timer();
	YWeatherParser ywp = new YWeatherParser();
	WeatherRecord lastRecord;
	WeatherTask curTask;
	
	long startupDelay;
	long period;
	
	static Logger logger = Logger.getLogger(WeatherChecker.class);

	
	public Vector<WeatherChangeListener> listeners= new Vector<WeatherChangeListener>();
	
	
	public void addListener(WeatherChangeListener listener) {
		listeners.add(listener);
	}

	public void removeListener(WeatherChangeListener listener) {
		listeners.remove(listener);
	}

	
	public WeatherChecker(long startupDelay, long period) {
		this.startupDelay = startupDelay;
		this.period = period;
	}
	
	
	
	public void stop() {
		if(curTask!=null) {
			curTask.cancel();
			curTask = null;
		}
	}
	
	public void start() {
		if(curTask!=null) {
			curTask.cancel();
		}
		curTask = new WeatherTask(); 
		TIMER.scheduleAtFixedRate(curTask, startupDelay, period);
	}

	
	
	public class WeatherTask extends TimerTask {

		public void run() {
			try { // we don't want this thread to die so catch any exception and keep going
				
				logger.info("fetching weather");
				
				WeatherRecord record = ywp.fetch();
				
				WeatherChangedEvent wce = WeatherChangedEvent.generate(lastRecord, record);
				
				if(wce != null) {
					for(WeatherChangeListener listener : listeners) {
						listener.weatherChanged(wce);
					}
				}
				
				
				lastRecord = record;

			

			} catch (Exception e) {
				logger.error("Exception fetching weather info\n" + e);
			}
		}
		
	}
}
