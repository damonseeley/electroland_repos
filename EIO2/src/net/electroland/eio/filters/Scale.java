package net.electroland.eio.filters;

import net.electroland.eio.Value;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class Scale implements Filter {

    private static Logger logger = Logger.getLogger(Scale.class);
    double scaleFactor;

    @Override
    public void configure(ParameterMap map) { 
        scaleFactor = map.getRequiredDouble("scaleFactor");
        logger.info("\t\tconfigured Scale for " + scaleFactor + " scaling.");
    }

    @Override
    public void filter(Value in) {
        in.setValue((int)(scaleFactor * in.getValue()));
    }
}