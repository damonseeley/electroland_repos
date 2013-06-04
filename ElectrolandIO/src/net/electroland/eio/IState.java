package net.electroland.eio;

import net.electroland.eio.filters.IOFilter;

public class IState extends IOState{

    boolean state;
    boolean lastState = false;
    long lastStateChange = System.currentTimeMillis();
    boolean isSuspect;
    int suspectThreshold;

    public IState(String id, double x, double y, double z, String units, int suspectThreshold) {
        super(id, x, y, z, units);
        this.suspectThreshold = suspectThreshold;
        this.isSuspect = false;
    }
    public void setState(boolean state)
    {
        for (IOFilter f : this.filters)
        {
            state = f.filter(state);
        }
        this.state = state;
        long duration = System.currentTimeMillis() - lastStateChange;
        if (state != lastState){
            lastState = state;
            lastStateChange = System.currentTimeMillis();
            duration = 0;
        }
                                                  // This iState is suspect:
        isSuspect =  suspectThreshold > 0 &&      // if a suspect threshold is specified,
                     state &&                     // and this IState is 'on',
                     duration > suspectThreshold; // and the threshold is surpassed.
    }

    /**
     * returns the state.  Defaults to 'OFF' if it is suspicious.
     * @return
     */
    public boolean getState()
    {
        return state;
    }
    /**
     * returns true if this iState has been on longer than the allowable
     * threshold for suspicious behavior.
     * 
     * @return
     */
    public boolean isSuspect() {
        return isSuspect;
    }
}