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
    private int avgListLength,tripLength;
    private int pm1CurTrips,pm2CurTrips;
    private long timeDomain,framerate;

    /**
     * This objects adds an event and time for every trip in the set of 
     * istates.  One can call getFlowData(millis) to get the int number
     * of trips in the millis domain provided
     */
    public TrafficFlowAnalyzer(long fr)
    {
        avgListLength = 50;
        tripLength = avgListLength;
        pm1trips = new ArrayList<Long>(tripLength);
        pm2trips = new ArrayList<Long>(tripLength);
        pm1AvgTrips = new ArrayList<Integer>(avgListLength);
        pm2AvgTrips = new ArrayList<Integer>(avgListLength);
        pm1Avg = 0;
        pm2Avg = 0;
        logger.info("TrafficFlowModelWatcher created");

        //default 30s
        timeDomain = 30000;
        framerate = fr;

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
        timeDomain = td;
        return pm1CurTrips;
    }

    public int getPM2Flow(int td) {
        // get flow in timedomain for PPLMVR #1
        timeDomain = td;
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
            try {
                for (long triptime : pm1trips)
                {
                    if (triptime > System.currentTimeMillis() - timeDomain) {
                        pm1CurTrips++;
                    }
                }
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }


            // get flow in timedomain for PPLMVR #1
            pm2CurTrips = 0;
            try {
                for (long triptime : pm2trips)
                {
                    if (triptime > System.currentTimeMillis() - timeDomain) {
                        pm2CurTrips++;
                    }
                }
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }





            // calc avg trips for PM1
            int trips = 0;
            try {
                for (long triptime : pm1trips)
                {
                    if (triptime > System.currentTimeMillis() - 30000) { // not sure this math is right
                        trips++;
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }       
            // should really clone subset of fixed size here for accuracy
            pm1AvgTrips.add(trips);
            int tripsToAvg = 0;
            int tripCount = 0;
            for (int tripAvg : pm1AvgTrips)
            {
                tripsToAvg += tripAvg;
                tripCount++;
            }
            pm1Avg = tripsToAvg/tripCount;



            // calc avg trips for PM2
            int trips2 = 0;
            try {
                for (long triptime : pm2trips)
                {
                    if (triptime > System.currentTimeMillis() - 30000) { // not sure this math is right
                        trips2++;
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
            pm2AvgTrips.add(trips2);
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
                //logger.info("TFA: pm1trips List trimmed");
                pm1trips.trimToSize();
            }
            if (pm2trips.size() > avgListLength) {
                //logger.info("TFA: pm2trips List trimmed");
                pm2trips.trimToSize();
            }

            if (pm1AvgTrips.size() > avgListLength) {
                //logger.info("TFA: pm1Avg List trimmed");
                pm1AvgTrips.trimToSize();
            }
            if (pm2AvgTrips.size() > avgListLength) {
                //logger.info("TFA: pm2Avg List trimmed");
                pm2AvgTrips.trimToSize();
            }



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