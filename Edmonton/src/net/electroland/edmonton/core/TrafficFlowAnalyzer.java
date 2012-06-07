package net.electroland.edmonton.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

import org.apache.log4j.Logger;

public class TrafficFlowAnalyzer {

	static Logger logger = Logger.getLogger(TrafficFlowAnalyzer.class);

	private ArrayList<Long> pm1trips;
	private ArrayList<Long> pm2trips;
	
    /**
     * This objects adds an event and time for every trip in the set of 
     * istates.  One can call getFlowData(millis) to get the int number
     * of trips in the millis domain provided
     */
    public TrafficFlowAnalyzer()
    {
        pm1trips = new ArrayList<Long>();
        pm2trips = new ArrayList<Long>();
        logger.info("TrafficFlowModelWatcher created");
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

    public int getPPM1Flow(int timeDomain) {
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
    
    public int getPPM2Flow(int timeDomain) {
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

}