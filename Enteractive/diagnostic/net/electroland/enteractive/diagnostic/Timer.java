package net.electroland.enteractive.diagnostic;

public class Timer extends Thread {
	boolean isBlocking = true; 
	private Object lock = new Object();
	protected long sleepTime;
	protected boolean isRunning = true;
	protected float framerate;
	
	protected float numErrors;
	
	public Timer(float framerate) {  
		setFrameRate(framerate);
		numErrors = 0;
	}
	
	public void setFrameRate(float framerate) {
		this.framerate = framerate;
		sleepTime = (long) (1000.0/framerate);
	}
	
	/*
	 * block until its time to re-run your frame
	 */
	public void block() {
		synchronized(lock) {
			isBlocking = true;
			try {
				lock.wait();
			} catch (InterruptedException e) {
			}
		}
	}

	protected void unblock() {
		synchronized(lock) {
			lock.notifyAll();
			if(! isBlocking) // other thread didn't have time to finish what it was doing a block before time ran out 
				numErrors++;
			if (numErrors > 100){
				//turned off for making me mad
				//System.err.println("WARNING: framerate " + framerate + " has been too high for timer 100 times");
				numErrors = 0;
			}
				
			isBlocking = false;
		}		
	}
	public void run() {
		while(isRunning) {
			unblock();
			synchronized(this) {
				try {
					wait(sleepTime);
				} catch (InterruptedException e) {
				}
			}
		}
		
	}
	
	public void stopRunning() {
		isRunning = false;
		synchronized(this) {
			notify();
		}
		synchronized(lock) {
			lock.notifyAll();
			isBlocking = false;
		}		
	}
}
