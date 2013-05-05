package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * This class implements a short-term-average over long-term-average filter, 
 * which can be used to detect sudden increases in signal energy. It employs 
 * simple one-pole low-pass filters to calculate both the short-term and long-term 
 * energy levels, and optionally delays the long-term average to lag behind the 
 * short-term average (which may allow more time for peaks to come through 
 * before the long-term average catches up). Additionally, minimum and maximum
 * values for the long-term average may be supplied to limit its ability to 
 * amplify and reduce the short-term average values. These limits are specified 
 * as normalized values in the interval [0.0 1.0], where the input data is assumed
 * to be within the range +-Short.MAX_VALUE/2.
 * 
 * Additionally, a "maxOutputVal" param can be used to specify how the resulting
 * floating-point sta / lta values are scaled and clipped before returning as a
 * fixed-point integer value. Generally, maxOutputVal should be set close to but
 * higher than the highest expected sta / lta value in order to avoid actually
 * clipping signal while retaining maximum precision in the unclipped range.
 * 
 * 
 * @author Sean
 *
 */

public class StaOverLtaFilter implements Filter {

    private final static int maxInputValAllowed = (int) Math.pow(Integer.MAX_VALUE, 0.5);
    
    private OnePoleLowpassFilter staFilt;
    private OnePoleLowpassFilter ltaFilt;
    private DelayLine ltaDelayLine;
    private int ltaMinVal, ltaMaxVal;
    private double maxOutputVal;
    
    @Override
    public void configure(ParameterMap params) {
        
        // Extract parameters
        double sta_tauInSamp = params.getDefaultDouble("sta_tauInSamples", 0.0);
        double sta_tauInSec = params.getDefaultDouble("sta_tauInSeconds", 0.0);
        
        double lta_tauInSamp = params.getDefaultDouble("lta_tauInSamples", 0.0);
        double lta_tauInSec = params.getDefaultDouble("lta_tauInSeconds", 0.0);

        double fs = params.getDefaultDouble("fs", 0.0);
        int ltaDelay = params.getDefaultInt("ltaDelay", 0);
        
        ltaMinVal = (int) ( Math.pow(Short.MAX_VALUE/2.0, 2) * params.getDefaultDouble("ltaMinVal", 0.0) );
        ltaMaxVal = (int) ( Math.pow(Short.MAX_VALUE/2.0, 2) * params.getDefaultDouble("ltaMaxVal", 1.0) );
        if (ltaMinVal == 0) // Set to 1 to avoid division by 0
            ltaMinVal = 1;
        
        maxOutputVal = params.getDefaultDouble("maxOutputVal", 100.0);
        
        
        // Validate parameter set
        if (sta_tauInSamp == 0.0 && sta_tauInSec == 0.0)
            throw new RuntimeException("The short-term-average time constant tau must be specified for an StaOverLtaFilter.");
        else if (lta_tauInSamp == 0.0 && lta_tauInSec == 0.0)
            throw new RuntimeException("The long-term-average time constant tau must be specified for an StaOverLtaFilter.");
        else if ((sta_tauInSamp == 0.0 || lta_tauInSamp == 0.0) && fs == 0.0)
            throw new RuntimeException("The sampling rate, fs, must be specified when supplying either tauInSeconds for an StaOverLtaFilter.");
        
        
        // Configure STA and LTA filters
        ParameterMap staParams = new ParameterMap();
        if (sta_tauInSamp == 0.0){
            staParams.put("tauInSeconds", Double.toString(sta_tauInSec));
            staParams.put("fs", Double.toString(fs));
        }
        else
            staParams.put("tauInSamples", Double.toString(sta_tauInSamp));
        
        staFilt = new OnePoleLowpassFilter();
        staFilt.configure(staParams);
        
        
        ParameterMap ltaParams = new ParameterMap();
        if (lta_tauInSamp == 0.0) {
            ltaParams.put("tauInSeconds", Double.toString(lta_tauInSec));
            ltaParams.put("fs", Double.toString(fs));
        }
        else
            ltaParams.put("tauInSamples", Double.toString(lta_tauInSamp));
        
        ltaFilt = new OnePoleLowpassFilter();
        ltaFilt.configure(ltaParams);
        
        
        // Configure LTA delay line
        ParameterMap delayLineParams = new ParameterMap();
        delayLineParams.put("delay", Integer.toString(ltaDelay));
        ltaDelayLine = new DelayLine();
        ltaDelayLine.configure(delayLineParams);
        
        // Initialize LTA delay line to hold ltaMaxVal to avoid division by 0  and suppress
        //    STA / LTA output during start-up
        ltaDelayLine.setAllValues(ltaMaxVal);
        
    }

    @Override
    public void filter(Value in) {
        
        // Because Value objects store data internally as int's, we have to watch out for overflow when
        //    we square below. If the input value is >= sqrt( Integer.MAX_VALUE ), overflow will occur.
        //    Here, we throw an exception rather than clip the value if this is going to happen (it may
        //    be best to know that this is happening and adjust system design to account for it). Note 
        //    that if in.getValue() is < Short.MAX_VALUE, we will never overflow here. Also, the staFilt
        //    and ltaFilt outputs will always be <= the square of the maximum input value (they are 
        //    averages with gain <= 1 at all frequencies), so if we don't overflow at the input, we will 
        //    never overflow at the output.
        if ( in.getValue() >= maxInputValAllowed )
            throw new RuntimeException("Overflow occurred in StaOverLtaFilter.");
        
        
        // Calculate current STA and LTA values
        Value staVal = new Value( (int) Math.pow(in.getValue(), 2) );
        Value ltaVal = new Value( staVal.getValue() );
        
        staFilt.filter(staVal);
        
        ltaDelayLine.filter(ltaVal);
        ltaFilt.filter(ltaVal);
        
        
        // Limit LTA value be within the interval [ltaMinVal ltaMaxVal]
        ltaVal.setValue( Math.min( Math.max(ltaVal.getValue(), ltaMinVal), ltaMaxVal ) );
        
        
        // Calculate sta / lta as a double and scale so that we get something useful when we convert back to int.
        //    We are generally interested in fairly small sta / lta values (0 to 100, for example), but we want to 
        //    retain detail over this range so we map the range of values from [0 maxOutputVal] to [0 Integer.MAX_VALUE],
        //    clipping sta / lta values that are > maxOutputVal.
        double output = Integer.MAX_VALUE / maxOutputVal * ((double) staVal.getValue()) / ((double) ltaVal.getValue());
        if (output > Integer.MAX_VALUE)
            output = Integer.MAX_VALUE;
        
        
        // Return final output value
    	in.setValue((int) output);
    }

}
