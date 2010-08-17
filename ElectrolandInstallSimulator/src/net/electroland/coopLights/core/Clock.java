package net.electroland.coopLights.core;

import java.util.Calendar;

public class Clock extends Thread {
	private static long ZEROHOUR;

	private static long MIDNIGHT;

	private Calendar startCal;

	private Calendar endCal;

	private boolean isUp = false;

	private boolean isRunning = true;

	private boolean isStartEndInverted = false;

	ClockHandler clockHandler;

	/**
	 * 
	 * @param startHour -
	 *            in 24 hour format
	 * @param startMin
	 * @param startSec
	 * @param endHour -
	 *            in 24 hour format
	 * @param endMin
	 * @param endSec
	 */

	public Clock(ClockHandler clockHandler, int startHour, int startMin,
			int startSec, int endHour, int endMin, int endSec) {
		if (MIDNIGHT == 0) { // only set once
			Calendar c = Calendar.getInstance();
			c.set(0, 0, 0, 0, 0, 0);
			ZEROHOUR = c.getTimeInMillis();
			c.add(Calendar.DATE, 1);
			MIDNIGHT = c.getTimeInMillis();
		}

		this.clockHandler = clockHandler;

		startCal = Calendar.getInstance();
		startCal.set(0, 0, 0, startHour, startMin, startSec);

		endCal = Calendar.getInstance();
		endCal.set(0, 0, 0, endHour, endMin, endSec);

		inversionCheck();

	}

	protected void inversionCheck() {
		if (endCal.before(startCal)) {// is ending in the AM (or starting in
										// the PM)
			isStartEndInverted = true;
		} else {
			isStartEndInverted = false;
		}

	}

	public void setStartTime(int h, int m, int s) {
		startCal = Calendar.getInstance();
		startCal.set(0, 0, 0, h, m, s);
		inversionCheck();
		synchronized (this) {
			notify();// wake and reset wait
		}
	}

	public void setEndTime(int h, int m, int s) {
		endCal = Calendar.getInstance();
		endCal.set(0, 0, 0, h, m, s);
		inversionCheck();
		synchronized (this) {
			notify();// wake and reset wait
		}
	}

	public static Calendar getCurTime() {
		Calendar curCal = Calendar.getInstance();
		curCal.set(0, 0, 0, curCal.get(Calendar.HOUR_OF_DAY), curCal.get(Calendar.MINUTE), curCal.get(Calendar.SECOND)); // zero
																		// out
																		// date
		return curCal;
	}

	public void run() {
		while (isRunning) {
			boolean oldState = isUp;

			determinUpState();
			
			System.out.println("oldstate is " + oldState);
			System.out.println("new state is " + isUp);

			if (oldState != isUp) {
				if (isUp) {
					System.out.println("calling start time");
					clockHandler.handleStartTime();
				} else {
					System.out.println("calling end time");
					clockHandler.handleEndTime();
				}
			}
			synchronized (this) {
				try {
					wait(getSleepTime());
				} catch (InterruptedException e) {
				}
			}
			System.out.println("clock waking");
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

	public void stopRunning() {
		isRunning = false;
		synchronized (this) {
			notify();
		}
	}

	public boolean isUp() {
		return isUp;
	}

	protected long getSleepTime() {
		Calendar curCal = getCurTime();
		if (curCal.before(startCal)) {
			if (isStartEndInverted && curCal.before(endCal)) {
				return endCal.getTimeInMillis() - curCal.getTimeInMillis() +1;
			} else {
				return startCal.getTimeInMillis() - curCal.getTimeInMillis() + 1;
			}
		} else if (curCal.before(endCal)) {
			return endCal.getTimeInMillis() - curCal.getTimeInMillis() +1;
		} else { // after everything
			if (isStartEndInverted) {
				return (MIDNIGHT - curCal.getTimeInMillis())
						+ (endCal.getTimeInMillis() - ZEROHOUR) +1;
			} else {
				return (MIDNIGHT - curCal.getTimeInMillis())
						+ (startCal.getTimeInMillis() - ZEROHOUR) + 1;
			}
		}
	}

	protected void determinUpState() {
		Calendar curCal = getCurTime();
		System.out.println("determinUpState:  isStartEndInverted=" + isStartEndInverted );
		System.out.println("                   startCal.before(curCal)=" + startCal.before(curCal));
		System.out.println("                   curCal.before(endCal)=" + curCal.before(endCal));
		if (isStartEndInverted) {
			isUp = startCal.before(curCal) || curCal.before(endCal);
		} else {
			isUp = startCal.before(curCal) && curCal.before(endCal);
		}
	}

	public static interface ClockHandler {
		public void handleStartTime();

		public void handleEndTime();
	}
	

}
