package net.electroland.edmonton.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

import net.electroland.ea.AnimationManager;
import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;
import net.electroland.utils.lighting.InvalidPixelGrabException;

import org.apache.log4j.Logger;

public class TrafficFlowAnalyzer extends Thread {

    static Logger logger = Logger.getLogger(TrafficFlowAnalyzer.class);

    private ArrayList<Long> pm1trips,pm2trips;
    private ArrayList<Integer> pm1AvgTrips,pm2AvgTrips;
    private long pm1Avg,pm2Avg;

    /**
     * This objects adds an event and time for every trip in the set of 
     * istates.  One can call getFlowData(millis) to get the int number
     * of trips in the millis domain provided
     */
    public TrafficFlowAnalyzer()
    {
        pm1trips = new ArrayList<Long>();
        pm2trips = new ArrayList<Long>();
        pm1AvgTrips = new ArrayList<Integer>();
        pm2AvgTrips = new ArrayList<Integer>();
        pm1Avg = 0;
        pm2Avg = 0;
        logger.info("TrafficFlowModelWatcher created");
        
        start();
    }

    public void trip(Collection<IState> states) {
        for (IState state : states) {
            if (state.getLocation().x > 340.0) {
                //trip was on PM1
                pm1trips.add(System.currentTimeMillis());
            } else if (state.getLocation().x < 245.0){   
                pm2trips.add(System.currentTimeMillis());
            }
        }
    }

    public int getPM1Flow(int timeDomain) {
        // get flow in timedomain for PPLMVR #1
        int tripHistory = 0;
        for (long triptime : pm1trips)
        {
            if (triptime > System.currentTimeMillis() - timeDomain) {
                tripHistory++;
            }
        }
        return tripHistory;
    }

    public int getPM2Flow(int timeDomain) {
        // get flow in timedomain for PPLMVR #1
        int tripHistory = 0;
        for (long triptime : pm2trips)
        {
            if (triptime > System.currentTimeMillis() - timeDomain) {
                tripHistory++;
            }
        }
        return tripHistory;
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

            // calc avg trips for PM1
            int trips = 0;
            for (long triptime : pm1trips)
            {
                if (triptime > System.currentTimeMillis() - 30000) { // not sure this math is right
                    trips++;
                }
            }
            //logger.info("PM1 TRIPS = " + trips);
            
            pm1AvgTrips.add(trips);
            int tripsToAvg = 0;
            int tripCount = 0;
            for (int tripAvg : pm1AvgTrips)
            {
                tripsToAvg += tripAvg;
                tripCount++;
            }
            pm1Avg = tripsToAvg/tripCount;
            //logger.info("PM1 TRIP AVG = " + pm1Avg);
            
            
            
            
            // calc avg trips for PM2
            int trips2 = 0;
            for (long triptime : pm2trips)
            {
                if (triptime > System.currentTimeMillis() - 30000) { // not sure this math is right
                    trips2++;
                }
            }
            //logger.info("PM1 TRIPS = " + trips);
            
            pm2AvgTrips.add(trips2);
            int tripsToAvg2 = 0;
            int tripCount2 = 0;
            for (int tripAvg : pm2AvgTrips)
            {
                tripsToAvg2 += tripAvg;
                tripCount2++;
            }
            pm2Avg = tripsToAvg2/tripCount2;
            //logger.info("PM1 TRIP AVG = " + pm1Avg);

            
            
            
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                logger.debug(e);
                e.printStackTrace();
            }
        }
    }










}