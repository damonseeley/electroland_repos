package net.electroland.norfolk.core;

public class FpsAverage {

	private int     data[];
    private int     index = 0;
    private double  total = 0;

	
	public FpsAverage(int samples){
		this.data = new int[samples];
	}
  
    public int getAverage(){
        return (int)(total/data.length);
    }
  
    // ring buffered sum
    public void add(int sample){
  
        total -= data[index];
        total += sample;
        data[index++] = sample;
  
        if (index == data.length){
            index = 0;
        }
    }	
}