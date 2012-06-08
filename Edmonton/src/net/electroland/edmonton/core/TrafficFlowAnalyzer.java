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

    //Thread stuff
    public static boolean isRunning;	
    private static float framerate;
    private static FrameTimer timer;

    private ArrayList<Long> pm1trips,pm2trips;
    private ArrayList<Integer> pm1AvgTrips,pm2AvgTrips;
    private int pm1Avg,pm2Avg;

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

        /******** Thread Setup ********/
        framerate = 1; //update averages once per second
        isRunning = true;
        timer = new FrameTimer(framerate);
        start();
        logger.info("TFA started up at framerate = " + framerate);
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

    public int getPM1Avg(){
        return pm1Avg;
    }
    
    public int getPM2Avg(){
        return pm2Avg;
    }

    /************************* Main Loop ******************************/
    /* (non-Javadoc)
     * @see java.lang.Thread#run()
     */
    public void run() {

        timer.start();

        while (isRunning) {

            // calc avg trips for PM1
            int trips = 0;
            for (long triptime : pm1trips)
            {
                if (triptime > System.currentTimeMillis() - 30000) { // not sure this math is right
                    trips++;
                }
            }
            logger.info("TRIPS = " + trips);
            
            pm1AvgTrips.add(trips);
            int tripsToAvg = 0;
            int tripCount = 0;
            for (int tripAvg : pm1AvgTrips)
            {
                tripsToAvg += tripAvg;
                tripCount++;
            }
            pm1Avg = tripsToAvg/tripCount;
           

            //TO-DO reset pm1AvgTrips after it grows to a certain size
            
            //Thread ops
            timer.block();
        }
    }










}