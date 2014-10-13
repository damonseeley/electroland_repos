package net.electroland.enteractive.scheduler;

import java.util.Timer;
import java.util.TimerTask;

import net.electroland.enteractive.core.EnteractiveMain;

import org.apache.log4j.Logger;

public class TimedEvent {
	public DateIterator dateIterator;
	protected TimedEventListener eventListener;
	protected TimerTask curTask;
	protected static final Timer TIMER = new Timer();
	protected Boolean isRunning = Boolean.TRUE;
	public int hour, minute, sec;
	
	static Logger logger = Logger.getLogger(TimedEvent.class);



	public TimedEvent(int hour, int minute, int sec, TimedEventListener eventListener) {
		this.hour = hour;
		this.minute = minute;
		this.sec = sec;
		this.eventListener = eventListener;
		dateIterator = new DateIterator(hour, minute, sec);
		synchronized(isRunning) {
			if(isRunning.booleanValue()) {
				curTask = new ReschedulingTimerTask();
				TIMER.schedule(curTask, dateIterator.next().getTime());
			}
		}
	}

	public void reschedule(int hour, int minute, int sec) {
		cancel();
		dateIterator = new DateIterator(hour, minute, sec);
		synchronized(isRunning) {
			isRunning = Boolean.TRUE;
			logger.info("rescheduled");
			curTask = new ReschedulingTimerTask();
			TIMER.schedule(curTask, dateIterator.next().getTime());
		}
	}

	public void notifyAndReschedule() {
		eventListener.timedEvent(this);
		synchronized(isRunning) {
			if(isRunning.booleanValue()) {
				curTask = new ReschedulingTimerTask();
				TIMER.schedule(curTask, dateIterator.next().getTime());
			}
		}
	}

	public void cancel() {
		synchronized(isRunning) {
			isRunning = Boolean.FALSE;
			curTask.cancel();
		}
	}


	private class ReschedulingTimerTask extends TimerTask {
		public void run() {
			notifyAndReschedule();
		}
	}
}
