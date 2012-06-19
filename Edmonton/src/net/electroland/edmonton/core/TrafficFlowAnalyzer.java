package net.electroland.edmonton.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import net.electroland.ea.AnimationManager;
import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;
import net.electroland.utils.lighting.InvalidPixelGrabException;

import org.apache.log4j.Logger;

public class TrafficFlowAnalyzer extends Thread {

	static Logger logger = Logger.getLogger(TrafficFlowAnalyzer.class);

	private List<Long> pm1trips,pm2trips;
	private List<Integer> pm1AvgTrips,pm2AvgTrips;
	private long pm1Avg,pm2Avg;
	private int avgListLength,tripLength;
	private int pm1CurTrips,pm2CurTrips;
	private long curAvgTime,framerate;

	/**
	 * This objects adds an event and time for every trip in the set of 
	 * istates.
	 */
	public TrafficFlowAnalyzer(long fr, int curAvg, int runAvg)
	{
		// some hacky math here, make the average lists lengths equal the length in frames
		// eg framerate = 1 and runAvg = 120000 or 2 minutes = 120*1 = 120
		// eg framerate = 10 and runAvg = 300000 or 5 minutes = 300*10 = 3000
		avgListLength = (int)(runAvg/1000 * fr);
		curAvgTime = curAvg;
		tripLength = avgListLength; // just make this super long, does not matter since we calc based on time
		
		pm1trips = Collections.synchronizedList(new ArrayList<Long>(tripLength));
		pm2trips = Collections.synchronizedList(new ArrayList<Long>(tripLength));
		pm1AvgTrips = Collections.synchronizedList(new ArrayList<Integer>(avgListLength));
		pm2AvgTrips = Collections.synchronizedList(new ArrayList<Integer>(avgListLength));
		pm1Avg = 0;
		pm2Avg = 0;
		logger.info("TrafficFlowModelWatcher created, avgListLength:" + avgListLength + " curAvgTime:" + curAvgTime);
		framerate = fr;
		start();
	}

	//cleaner
	public void trip(double xloc) {
		if (xloc > 340.0) {
			//trip was on PM1
			pm1trips.add(System.currentTimeMillis());
		} else if (xloc < 245.0){   
			pm2trips.add(System.currentTimeMillis());
		}
	}

	public int getPM1Flow(int td) {
		curAvgTime = td;
		return pm1CurTrips;
	}

	public int getPM2Flow(int td) {
		// get flow in timedomain for PPLMVR #1
		curAvgTime = td;
		return pm2CurTrips;
	}

	public long getPM1Avg(){
		return pm1Avg;
	}

	public long getPM2Avg(){
		return pm2Avg;
	}

	/************************* Main Loop ******************************/
	/* (non-Javadoc)
	 * @see java.lang.Thread#run()
	 */
	public void run() {

		while (true) {

			// get flow in timedomain for PPLMVR #1
			pm1CurTrips = 0;
			for (long triptime : pm1trips)
			{
				if (triptime > System.currentTimeMillis() - curAvgTime) {
					pm1CurTrips++;
				}
			}

			// get flow in timedomain for PPLMVR #2
			pm2CurTrips = 0;
			for (long triptime : pm2trips)
			{
				if (triptime > System.currentTimeMillis() - curAvgTime) {
					pm2CurTrips++;
				}
			}
			
			
			// calc avg trips for PM1
			pm1AvgTrips.add(pm1CurTrips);
			int tripsToAvg = 0;
			int tripCount = 0;
			for (int tripAvg : pm1AvgTrips)
			{
				tripsToAvg += tripAvg;
				tripCount++;
			}
			pm1Avg = tripsToAvg/tripCount;



			// calc avg trips for PM2
			pm2AvgTrips.add(pm2CurTrips);
			int tripsToAvg2 = 0;
			int tripCount2 = 0;
			for (int tripAvg : pm2AvgTrips)
			{
				tripsToAvg2 += tripAvg;
				tripCount2++;
			}
			pm2Avg = tripsToAvg2/tripCount2;



			
			
			// TRIMMING
			// trim the lists if longer than they should be.
			
			if (pm1trips.size() > avgListLength) {
				int count = pm1trips.size() - avgListLength;
				for (int i=0; i<count; i++){
					pm1trips.remove(pm1trips.size()-1);
				}
			}
			if (pm2trips.size() > avgListLength) {
				//logger.info("TFA: pm2trips List trimmed");
				int count = pm2trips.size() - avgListLength;
				for (int i=0; i<count; i++){
					pm2trips.remove(pm2trips.size()-1);
				}

			}

			if (pm1AvgTrips.size() > avgListLength) {
				//logger.info("TFA: pm1AvgTrips List size = " + pm1AvgTrips.size());
				int count = pm1AvgTrips.size() - avgListLength;
				for (int i=0; i<count; i++){
					pm1AvgTrips.remove(pm1AvgTrips.size()-1);
				}
			}
			if (pm2AvgTrips.size() > avgListLength) {
				//logger.info("TFA: pm2AvgTrips List size = " + pm2AvgTrips.size());
				int count = pm2AvgTrips.size() - avgListLength;
				for (int i=0; i<count; i++){
					pm2AvgTrips.remove(pm2AvgTrips.size()-1);
				}
			}
			
			logger.info("TFA stats: pm1trips:" + pm1trips.size() + " pm1AvgTrips:" + pm1AvgTrips.size() + " pm2trips:" + pm2trips.size() + " pm2AvgTrips:" + pm2AvgTrips.size());



			try {
				Thread.sleep(1000/framerate);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				logger.debug(e);
				e.printStackTrace();
			}
		}
	}










}