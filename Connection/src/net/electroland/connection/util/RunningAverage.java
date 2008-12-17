package net.electroland.connection.util;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * @author geilfuss
 */
public class RunningAverage {

	private int maxDatapoints;
	private long maxDatapointAge;
	private boolean isIterationAverager = true;
	private ArrayList <Datapoint> vals;

	/**
	 * iterations is the total number of datapoints to store and average at any
	 * point in time.
	 * 
	 * @param iterations
	 */
	public RunningAverage(int maxDatapoints){
		this.maxDatapoints = maxDatapoints;
		isIterationAverager = true;
		vals = new ArrayList<Datapoint>();
	}

	/**
	 * millis is a duration value to calculate a running average for.
	 * 
	 * @param millis
	 */
	public RunningAverage(long ageInMillis){
		this.maxDatapointAge = ageInMillis;
		isIterationAverager = true;
		vals = new ArrayList<Datapoint>();
	}

	/**
	 * returns true if this Averager is averaging a limited number of iterations,
	 * not a limited duration.
	 * 
	 * @return
	 */
	public boolean isIterationAverager() {
		return isIterationAverager;
	}

	/**
	 * 
	 * @param value
	 */
	public void addValue(Double value){
		synchronized (vals){			
			vals.add(new Datapoint(System.currentTimeMillis(), value));
		}
	}

	/**
	 * add a value ONLY if millis milliseconds have passed since the last value
	 * was added.
	 * 
	 * @param value
	 * @param millis
	 */
	public void addValue(Double value, long millis){
		synchronized (vals){
			// check to see if enough time has elapsed.
			long curr = System.currentTimeMillis();
			if (vals.size() == 0){
				// if there are no other values, no need to test.
				vals.add(new Datapoint(curr, value));
			}else{
				if (curr - vals.get(vals.size() - 1).timestamp >= millis){
					vals.add(new Datapoint(curr, value));
				}
			}
		}
	}
	
	//TBD
	public double getFPS() throws NoDataException{
		return -1;
	}
	
	
	/**
	 * 
	 * @return
	 * @throws NoDataException
	 */
	public double getAvg() throws NoDataException{
		double average = 0;
		double points = 0;

		if (isIterationAverager){
			// pare down the size of the list to exactly as many values as
			// the iterator.
			Iterator <Datapoint> i;
			synchronized (vals){				
				while (vals.size() > maxDatapoints){
					vals.remove(0);
				}			
				i = vals.iterator();
			}
			while (i.hasNext()){
				Datapoint d = i.next();
				average += d.value;
				points += 1;
			}
		}else{
			// remove any values that are too old. calculate the average as it 
			// through the list.
			long time = System.currentTimeMillis();
			Iterator <Datapoint> i = vals.iterator();
			synchronized (vals){
				while (i.hasNext()){
					Datapoint d = i.next();
					if (time - d.timestamp <= maxDatapointAge){
						average += d.value;
						points += 1;
					}else{
						// remove as many as we don't read.
						vals.remove(0);
					}
				}
			}
		}
		if (points == 0){
			throw new NoDataException();
		}else{
			return average / points;
		}
	}

	/**
	 * 
	 * @return
	 */
	public int getMaxDatapoints() {
		return maxDatapoints;
	}

	/**
	 * 
	 * @return
	 */
	public long getMaxDatapointAge() {
		return maxDatapointAge;
	}

	public String toString(){
		try{
			StringBuffer sb = new StringBuffer("RunningAverage: ");
			sb.append(this.getAvg()).append("[");
			Iterator <Datapoint>i = vals.iterator();
			while (i.hasNext()){
				Datapoint dr = i.next();
				sb.append('[');
				sb.append(dr.timestamp).append(", ").append(dr.value);
				sb.append(']');
			}
			sb.append(']');
			return sb.toString();			
		}catch(NoDataException e){
			return "No Data in RunningAverage.";
		}
	}
	
	/**
	 * private class to hold a record for inclusinon in a running average.  
	 * simply a time-stamped datapoint.
	 * 
	 * @author geilfuss
	 *
	 */
	class Datapoint{
		public Datapoint(long timestamp, double value){
			this.timestamp = timestamp;
			this.value = value;
		}
		long timestamp;
		double value;
	}
}