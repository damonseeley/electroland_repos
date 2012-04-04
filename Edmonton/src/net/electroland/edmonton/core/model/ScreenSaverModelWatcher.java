package net.electroland.edmonton.core.model;

import java.util.Map;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

import org.apache.log4j.Logger;

/**
 * Watches overall activity.  In the event that the system either goes quiet or
 * becomes active, Listeners will receive an event. 
 * 
 * Going active is defined as having any sensor event after having been quiet.
 * 
 * Going quiet is defined as having no activiy for at least timeOut period.
 * 
 * You can set the timeOut using setTimeOut.  The default is 1 minute.
 * 
 * If you receive an event from this watcher, you can determine if it was an
 * active or quiet by calling evt.getSource().isQuiet(), where evt is the Event
 * returned to the listener.
 * 
 * @author bradley
 *
 */
public class ScreenSaverModelWatcher extends ModelWatcher {

    static Logger logger = Logger.getLogger(ScreenSaverModelWatcher.class);

    private boolean isQuiet = true;
    private long lastActivity = 0;
    private long timeOut = 1000 * 60;

    public boolean isQuiet() {
        return isQuiet;
    }

    public void setTimeOut(long timeOut) {
        this.timeOut = timeOut;
    }

    @Override
    public boolean poll() {

        boolean anyAction = false;

        for (IState state : this.getStates())
        {
            anyAction = anyAction || (!state.isSuspect() && state.getState());
        }

        if (anyAction){
            lastActivity = System.currentTimeMillis();
            if (isQuiet){
                isQuiet = false;
                return true;
            }
        }else{
            if (!isQuiet && System.currentTimeMillis() - lastActivity > timeOut)
            {
                isQuiet = true;
                return true;
            }
        }

        return false;
    }

    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        return null;
    }
}