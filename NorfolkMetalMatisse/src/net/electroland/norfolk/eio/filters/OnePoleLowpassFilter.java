package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * This class implements a simple one-pole low-pass filter (which has an impulse 
 * response that follows an exponential decay). The time constant for the filter
 * may be supplied in terms of samples as "tauInSamples", or in terms of seconds
 * as "tauInSeconds" (in which case the current sampling rate must also be 
 * supplied as the parameter "fs". Note that it's ok to specify fractional sample 
 * amounts.
 * 
 * @author Sean
 *
 */

public class OnePoleLowpassFilter implements Filter {

    private double b0;
    // Assume a0 == 1
    private double a1;
    private double z1;
    
    @Override
    public void configure(ParameterMap params) {
        
        // Extract parameters
        double tauInSamp = params.getDefaultDouble("tauInSamples", 0.0);
        
        double tauInSec = params.getDefaultDouble("tauInSeconds", 0.0);
        double fs = params.getDefaultDouble("fs", 0.0);
        
        // Validate parameter set
        if (tauInSamp == 0.0 && tauInSec == 0.0)
            throw new RuntimeException("The time constant tau must be specified for a one-pole low-pass filter.");
        else if (tauInSamp == 0.0 && fs == 0.0)
            throw new RuntimeException("The sampling rate, fs, must be specified when supplying tauInSeconds for a one-pole low-pass filter.");
        
        // Calculate tau
        double tau;
        if (tauInSamp == 0.0)
            tau = tauInSec * fs;
        else
            tau = tauInSamp;
        
        // Calculate filter coefficients
        double coeff = Math.exp( -1/tau );
        b0 = 1 - coeff;
        a1 = -coeff;
        
        // Initialize state to 0
        z1 = 0.0;
    }

    @Override
    public void filter(Value in) {
        
        double input = in.getValue();
        
        double output = z1 + b0 * input;
        z1 = -a1 * output;
        
    	in.setValue((int) output);
    }

}
