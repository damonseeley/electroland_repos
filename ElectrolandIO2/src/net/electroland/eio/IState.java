package net.electroland.eio;

import net.electroland.eio.filters.IOFilter;

import org.apache.log4j.Logger;

public class IState extends IOState{

    private static Logger logger = Logger.getLogger(IState.class);    
    boolean state;
    boolean lastState = false;
    long lastStateChange = System.currentTimeMillis();
    
    public IState(String id, int x, int y, int z, String units) {
        super(id, x, y, z, units);
    }
    public void setState(boolean state)
    {
        for (IOFilter f : this.filters)
        {
            state = f.filter(state);
        }
        this.state = state;
        if (state != lastState){
        	lastState = state;
        	long duration = System.currentTimeMillis() - lastStateChange;
        	lastStateChange = System.currentTimeMillis();
        	logger.info("IOState." + id + " switched to: " + state + " after " + duration + " millis.");
        }
    }
    public boolean getState()
    {
        return state;
    }
}
