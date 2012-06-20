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
    private List<Integer> pm1MovingAvgTrips,pm2MovingAvgTrips;
    private long pm1Avg,pm2Avg;
    private int avgListLength,tripLength;
    private int pm1LocalTrips,pm2LocalTrips;
    private long curAvgTime,framerate;
    private long starttime,reporttime;

    /**
     * This objects adds an event and time for every trip in the set of 
     * istates.
     * @fr = frame rate
     * @curAvg = the local average domain in ms
     * @runAvg = the time domain for a moving average in ms
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
        pm1MovingAvgTrips = Collections.synchronizedList(new ArrayList<Integer>(avgListLength));
        pm2MovingAvgTrips = Collections.synchronizedList(new ArrayList<Integer>(avgListLength));
        pm1Avg = 0;
        pm2Avg = 0;
        starttime = System.currentTimeMillis();
        reporttime = starttime;
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

    public int getPM1Flow() {
        return pm1LocalTrips;
    }

    public int getPM2Flow() {
        return pm2LocalTrips;
    }

    public long getPM1Avg(){
        return pm1Avg;
    }

    public long getPM2Avg(){
        return pm2Avg;
    }

    public void logpm1() {
        try {
            for (long triptime : pm1trips)
            {
                logger.info(triptime);
            }
        } catch (Exception e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace();
        }
    }

    /************************* Main Loop ******************************/
    /* (non-Javadoc)
     * @see java.lang.Thread#run()
     */
    public void run() {

        while (true) {

            // get flow in timedomain for PPLMVR #1
            pm1LocalTrips = 0;
            try {
                for (long triptime : pm1trips)
                {
                    if (triptime > System.currentTimeMillis() - curAvgTime) {
                        pm1LocalTrips++;
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }

            // get flow in timedomain for PPLMVR #2
            pm2LocalTrips = 0;
            try {
                for (long triptime : pm2trips)
                {
                    if (triptime > System.currentTimeMillis() - curAvgTime) {
                        pm2LocalTrips++;
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }


            // add the current trips/local timedomain to the moving avg list
            // calc moving avg for PM1 by adding all local averages together and dividing
            int tripsToAvg;
            tripsToAvg = 0;
            try {
                pm1MovingAvgTrips.add(pm1LocalTrips);
                for (int tripAvg : pm1MovingAvgTrips)
                {
                    tripsToAvg += tripAvg;
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
            pm1Avg = tripsToAvg/pm1MovingAvgTrips.size();



            // add the current trips/local timedomain to the moving avg list
            // calc moving avg for PM2 by adding all local averages together and dividing
            int tripsToAvg2;
            tripsToAvg2 = 0;
            try {
                pm2MovingAvgTrips.add(pm2LocalTrips);
                for (int tripAvg : pm2MovingAvgTrips)
                {
                    tripsToAvg2 += tripAvg;
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
            pm2Avg = tripsToAvg2/pm2MovingAvgTrips.size();





            // TRIMMING
            // trim the lists if longer than they should be.

            try {
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

                if (pm1MovingAvgTrips.size() > avgListLength) {
                    //logger.info("TFA: pm1AvgTrips List size = " + pm1AvgTrips.size());
                    int count = pm1MovingAvgTrips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm1MovingAvgTrips.remove(pm1MovingAvgTrips.size()-1);
                    }
                }
                if (pm2MovingAvgTrips.size() > avgListLength) {
                    //logger.info("TFA: pm2AvgTrips List size = " + pm2AvgTrips.size());
                    int count = pm2MovingAvgTrips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm2MovingAvgTrips.remove(pm2MovingAvgTrips.size()-1);
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }



            if ((System.currentTimeMillis() - reporttime) > 10000) {
                long timeElapsed = (System.currentTimeMillis() - starttime)/1000;
                logger.info("TFA LIST STATS: pm1trips:" + pm1trips.size() + " pm1AvgTrips:" + pm1MovingAvgTrips.size() + " pm2trips:" + pm2trips.size() + " pm2AvgTrips:" + pm2MovingAvgTrips.size());
                logger.info("TFA STATS: pm1SMA: " + pm1Avg + " pm2SMA: " + pm2Avg);
                logger.info("TIME SINCE APP START = " + timeElapsed/60 + "m " + timeElapsed%60 + "s");
                reporttime = System.currentTimeMillis();
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