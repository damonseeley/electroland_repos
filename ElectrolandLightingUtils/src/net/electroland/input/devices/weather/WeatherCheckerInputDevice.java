package net.electroland.input.devices.weather;

import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

import net.electroland.input.InputDeviceListener;

public class WeatherCheckerInputDevice  {
	protected static final Timer TIMER = new Timer();
	YWeatherParser ywp = new YWeatherParser();
	WeatherRecord lastRecord;
	WeatherTask curTask;
	
	long startupDelay;
	long period;
	
	public Vector<InputDeviceListener> listeners= new Vector<InputDeviceListener>();
	
	
	public void addListener(InputDeviceListener listener) {
		listeners.add(listener);
	}

	public void removeListener(InputDeviceListener listener) {
		listeners.remove(listener);
	}

	
	public WeatherCheckerInputDevice(long startupDelay, long period) {
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
				Date now = new Date();
				//DateFormat df = DateFormat.getDateInstance();
				// hacky change made to prevent number format exception in weather checker.
				System.out.println(now.toLocaleString()+" fetching weather");
				
				WeatherRecord record = ywp.fetch();
				
				WeatherChangedEvent wce = WeatherChangedEvent.generate(lastRecord, record);
				
				if(wce != null) {
					for(InputDeviceListener listener : listeners) {
						listener.inputReceived(wce);
					}
				}
				
				
				lastRecord = record;

			

			} catch (Exception e) {
				System.out.println("Exception fetching weather info\n" + e);
			}
		}
		
	}
}
