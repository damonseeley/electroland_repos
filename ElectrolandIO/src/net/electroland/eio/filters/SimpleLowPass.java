package net.electroland.eio.filters;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class SimpleLowPass implements IOFilter {

    private static Logger logger = Logger.getLogger(IOFilter.class);
    private int millis;

    @Override
    public void configure(ParameterMap params) {
        millis = params.getRequiredInt("millis");
        logger.info("\t\tconfigured SimpleLowPass filter for " + millis + " millis.");
    }

    @Override
    public byte filter(byte b) {
        return b;
    }

    @Override
    public boolean filter(boolean b) {
        return b;
    }
}