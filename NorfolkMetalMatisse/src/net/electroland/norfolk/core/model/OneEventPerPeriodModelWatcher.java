package net.electroland.norfolk.core.model;

import java.util.Map;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

import org.apache.log4j.Logger;

public class OneEventPerPeriodModelWatcher extends ModelWatcher {

	static Logger logger = Logger.getLogger(OneEventPerPeriodModelWatcher.class);

    private long periodMillis, timeSinceFired = 0;
    private String clipName;

    /**
     * This watcher will watch one or a set of IStates to see if anything turns
     * on.  If so, it announces the ON event.  It will not allow any more on 
     * events until a user-specified delay has passed since the ON event
     * occurred.
     * 
     * @param periodMillis - milliseconds that must elapse before another 
     * event can occur.  The period starts when the first event occurs.
     */
    public OneEventPerPeriodModelWatcher(long periodMillis)
    {
        this.periodMillis = periodMillis;
    }

    public OneEventPerPeriodModelWatcher(String clipName, long periodMillis)
    {
    	this.clipName 		= clipName;
        this.periodMillis 	= periodMillis;
    }


    public String getClipName() {
		return clipName;
	}

	public void setClipName(String clipName) {
		this.clipName = clipName;
	}

	@Override
    public boolean poll() {

        // see if ANY state I'm assigned to returns as being ON
        for (IState state : this.getStates())
        {
            // yes?
            if (!state.isSuspect() && state.getState())
            {
                // check to see if we fired an event too recently
                if (System.currentTimeMillis() - timeSinceFired > periodMillis)
                {
                    // if not, note the time
                    timeSinceFired = System.currentTimeMillis();
                    // and let the Model know that we saw something.
                    // In this case, we're not passing any optional details.
                    return true;
                } else {
                	//logger.info("OneEventPerPeriodModelWatcher: not enough time elapsed to fire");
                }
            }
        }
        // nothing to see here.
        return false;
    }

    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        // DO NOTHING.
        return null;
    }
}