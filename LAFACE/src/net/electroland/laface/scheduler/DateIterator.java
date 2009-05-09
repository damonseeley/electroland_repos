package net.electroland.laface.scheduler;

import java.text.DateFormat;
import java.util.Calendar;
import java.util.Iterator;

public class DateIterator implements Iterator<Calendar> {


	public Calendar lastDisplay;

	public DateIterator(int hour, int minute, int second) {
	
		Calendar curTime = Calendar.getInstance();
		Calendar compareTime = (Calendar) curTime.clone();
		
		compareTime.set(Calendar.HOUR_OF_DAY, hour);
		compareTime.set(Calendar.MINUTE, minute);
		compareTime.set(Calendar.SECOND, second);

		
		if(curTime.before(compareTime)) {
			compareTime.add(Calendar.DAY_OF_MONTH, -1);
		}
		lastDisplay = compareTime;
	}
	                                            
	
	public String toString() {
		 return  DateFormat.getDateTimeInstance(DateFormat.LONG, DateFormat.LONG).format(lastDisplay.getTime());
		  
	}
	public boolean hasNext() {
		return true;
	}

	public Calendar next() {
		lastDisplay.add(Calendar.DAY_OF_MONTH, 1);
		return lastDisplay;
	}
	
	public Calendar current() {
		return lastDisplay;
	}

	public void remove() {
	}
	
	
	
}
