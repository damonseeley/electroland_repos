package net.electroland.norfolk.core;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.log4j.Logger;

public class TrafficFlowAnalyzer extends Thread {

    static Logger logger = Logger.getLogger(TrafficFlowAnalyzer.class);

    private Queue<Long> pm1trips,pm2trips;
    private Queue<Integer> pm1MovingAvgTrips,pm2MovingAvgTrips;
    private long pm1Avg,pm2Avg;
    private int avgListLength;
    private int pm1LocalTrips,pm2LocalTrips;
    private long curAvgTime,runAvgTime,framerate;
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
        runAvgTime = runAvg;

        pm1trips          = new ConcurrentLinkedQueue<Long>();
        pm2trips          = new ConcurrentLinkedQueue<Long>();
        pm1MovingAvgTrips = new ConcurrentLinkedQueue<Integer>();
        pm2MovingAvgTrips = new ConcurrentLinkedQueue<Integer>();

        pm1LocalTrips = 0;
        pm2LocalTrips = 0;
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
    
    public long getCurAvgTime() {
        return curAvgTime;
    }
    
    public long getRunAvgTime() {
        return runAvgTime;
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
            synchronized(pm1MovingAvgTrips)
            {
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
            }



            // add the current trips/local timedomain to the moving avg list
            // calc moving avg for PM2 by adding all local averages together and dividing
            int tripsToAvg2;
            tripsToAvg2 = 0;
            synchronized(pm2MovingAvgTrips)
            {
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
            }




            // TRIMMING
            // trim the lists if longer than they should be.
            // note trimming should be from the FRONT of the list

            try {
                
                if (pm1trips.size() > avgListLength) {
                    int count = pm1trips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm1trips.remove();
                    }
                }
                if (pm2trips.size() > avgListLength) {
                    //logger.info("TFA: pm2trips List trimmed");
                    int count = pm2trips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm2trips.remove();
                    }
                }

                if (pm1MovingAvgTrips.size() > avgListLength) {
                    //logger.info("TFA: pm1AvgTrips List size = " + pm1AvgTrips.size());
                    int count = pm1MovingAvgTrips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm1MovingAvgTrips.remove();
                    }
                }
                if (pm2MovingAvgTrips.size() > avgListLength) {
                    //logger.info("TFA: pm2AvgTrips List size = " + pm2AvgTrips.size());
                    int count = pm2MovingAvgTrips.size() - avgListLength;
                    for (int i=0; i<count; i++){
                        pm2MovingAvgTrips.remove();
                    }
                }
            } catch (Exception e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }



            if ((System.currentTimeMillis() - reporttime) > 60000) {
                int curAvgS = (int)(curAvgTime/1000);
                int runAvgS = (int)(runAvgTime/1000);
                //logger.info("TFA LIST STATS: pm1trips:" + pm1trips.size() + " pm1AvgTrips:" + pm1MovingAvgTrips.size() + " pm2trips:" + pm2trips.size() + " pm2AvgTrips:" + pm2MovingAvgTrips.size());
                logger.info("TFA STATS: pm1\t" + curAvgS + "s:\t" + pm1LocalTrips + "\tpm1 " + runAvgS + "s SMA:\t" + pm1Avg + "\tpm2 " + curAvgS + "s:\t" + pm2LocalTrips + "\tpm2 " + runAvgS + "s SMA:\t" + pm2Avg);
                //logger.info("TIME SINCE APP START = " + timeElapsed/60 + "m " + timeElapsed%60 + "s");
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