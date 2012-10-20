package net.electroland.elvis.util;

import java.util.Vector;

public class TimeOutMonitorThread extends Thread {
	Vector<TimeOutListener> listeners = new Vector<TimeOutListener>();
	long timeOutTime;	
	long timeOutDuration;
	boolean isRunning = true;
	String timeOutMonitorName;
	public TimeOutMonitorThread(long timeOutDuration) {
		this(timeOutDuration, "");
	}

	public TimeOutMonitorThread(long timeOutDuration, String timeOutMonitorName) {
		this.timeOutMonitorName = timeOutMonitorName;
		this.timeOutDuration = timeOutDuration;
		timeOutTime = -1;
	}


	public void stopRunning() {
		isRunning = false;
		synchronized(this) {
			this.notifyAll();
		}
	}

	public void run() {
		// lets wait twice as long in the beginning to wait for initialization

		timeOutTime = System.currentTimeMillis() + 2* timeOutDuration;
		while(isRunning) {
			long sleepTime =timeOutTime -  System.currentTimeMillis();
			if(sleepTime > 0) {
				synchronized(this) {
					try {
						this.wait(sleepTime);
					} catch (InterruptedException e) {
					}
				}
			} else {
				for(TimeOutListener tol : listeners) {
					tol.monitorTimedOut(this);
					isRunning = false;
				}
			}
		}

	}
	public void updateTimeOut() {
		timeOutTime = System.currentTimeMillis() + timeOutDuration;
	}

	public void addTimeOutListener(TimeOutListener tol) {
		listeners.add(tol);
	}

	public static interface TimeOutListener {
		public void monitorTimedOut(TimeOutMonitorThread sender);

	}

}
