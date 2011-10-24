package net.electroland.eio.filters;

import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class DelayedOffFilter implements IOFilter {

    private static Logger logger = Logger.getLogger(DelayedOffFilter.class);
    private long millis, laston;

    @Override
    public void configure(ParameterMap params) {
        millis = params.getRequiredInt("millis");
        logger.info("\t\tconfigured DelayedOffFilter for " + millis + " millis.");
    }

    @Override
    public byte filter(byte b) {
        throw new RuntimeException("This filter hasn't been enabled for bytes yet.");
    }

    @Override
    public boolean filter(boolean b) {
        long current = System.currentTimeMillis();
        if (b){
            laston = current;
            return true;
        }else{
            return current - laston < millis;
        }
    }

}
