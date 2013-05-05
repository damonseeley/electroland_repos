package net.electroland.norfolk.core;

public class FpsAverage {

	private int     data[];
    private int     index = 0;
    private float  total = 0;
    private long lastTouch = 0;

	
	public FpsAverage(int samples){
		this.data = new int[1];
	}
  
    public float getAverage(){
        return (1000.0f)/(total / data.length);
    }
  
    // ring buffered sum
    public void touch(){
  
        long current    = System.currentTimeMillis();
        int sample      = (int)(current - lastTouch);
        lastTouch       = current;

        total -= data[index];
        total += sample;
        data[index++] = sample;
  
        if (index == data.length){
            index = 0;
        }
    }	
}