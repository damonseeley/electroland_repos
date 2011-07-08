package net.electroland.utils;

public class FrameRateRingBuffer {

	long[] times;
	int pointer;
	long last;
	boolean firstPass = true;

	//  test
	public static void main(String args[])
	{
		FrameRateRingBuffer r = new FrameRateRingBuffer(10);
		for (int i = 0; i < 100; i++){
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			r.markFrame();
			System.out.println(r);
			System.out.println(r.getFPS());
		}
	}
	
	public FrameRateRingBuffer(int expectedFrameRate, int seconds)
	{
		times = new long[expectedFrameRate * seconds];
		last = System.currentTimeMillis();		
	}
	
	
	public FrameRateRingBuffer(int capacity){
		if (capacity < 1){
			throw new RuntimeException("capacity must be greater than 0.");
		}
		times = new long[capacity];
		last = System.currentTimeMillis();
	}
	
	public void markFrame()
	{
		synchronized(times){
			long curr = System.currentTimeMillis();
			times[pointer++] = curr - last;
			last = curr;
			if (pointer == times.length){
				pointer = 0;
				firstPass = false;
			}
		}
	}
	
	public double getFPS()
	{
		synchronized(times){
			long total = 0;
			for (long t : times){
				total += t;
			}
			return 1000 / 
					(total / 
						(firstPass ? (double)pointer : (double)times.length));
		}
	}

	public String toString(){
		StringBuffer sb = new StringBuffer("StringBuffer[fps=");
		sb.append(this.getFPS()).append(", times=[");
		for (long t : times){
			sb.append(t).append(',');
		}
		sb.append("]]");
		return sb.toString();
	}
}