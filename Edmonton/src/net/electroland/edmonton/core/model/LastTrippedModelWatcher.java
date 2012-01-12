package net.electroland.edmonton.core.model;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

/**
 * This watcher just records the last time any IState was tripped. You can
 * access the times by getting LastTrippedModelWatcher.TRIP_TIMES from the map 
 * returned by getOptionalPositiveDetails().
 * 
 * @author bradley
 *
 */
public class LastTrippedModelWatcher extends ModelWatcher {

    final public static String TRIP_TIMES = "trip_times";

    private Map<IState, Long> records 
        = Collections.synchronizedMap(new HashMap<IState, Long>());

    private Map<String, Object> details 
        = new HashMap<String, Object>();

    @Override
    public boolean poll() {

        if (details.isEmpty()){
            details.put(TRIP_TIMES, records);
        }

        for (IState state : this.getStates())
        {
            if (state.getState())
            {
                records.put(state, System.currentTimeMillis());
            }
        }
        return true;
    }

    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        return details;
    }
}