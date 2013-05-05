package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * This class implements a generic hold-off or time-out functionality that takes 
 * into account the values of a signal. A minimum hold-off time is specified,
 * and the filter then counts up to this amount of time but decrements or resets 
 * the counter whenever signal values above specified thresholds are encountered.
 * Thus, this class can be used to implement a hold-off which will wait until no
 * significant signal has been observed for a period of time equal to the minimum
 * hold-off time before declaring the hold-off period over.
 * 
 * By default, the counter penalty and reset thresholds are both set to 1.0 so 
 * that the class implements a normal hold-off based only on time. The filter() 
 * function will set the input Value to 1 if the hold-off has passed and 0 if the
 * hold-off is still active.
 * 
 * startHoldoff() must be called at the beginning of a hold-off period to initialize
 * the HoldoffFilter state.
 * 
 * This class assumes that input is some positive detection signal that has been 
 * scaled to the range [0 Integer.MAX_VALUE]. All threshold values are defined
 * on the normalized scale [0.0 1.0], corresponding to [0 Integer.MAX_VALUE].
 * 
 * 
 * @author Sean
 *
 */

public class HoldoffFilter_Debug implements Filter {

    private double counter, holdoffLenMs, currHoldoffLenMs;
    private double maxHoldoffCounter, maxHoldoffLenMs;
    
    private long prevTimeMs;
    
    private double penaltyThresh, resetThresh;
    private double penaltyMult, penaltyPow;

    @Override
    public void configure(ParameterMap params) {
        
        // Extract parameters
        holdoffLenMs = params.getRequiredDouble("holdoffLenMs");
        maxHoldoffLenMs = params.getDefaultDouble("maxHoldoffLenMs", Double.POSITIVE_INFINITY);
        penaltyThresh = params.getDefaultDouble("penaltyThresh", 1.0);
        resetThresh = params.getDefaultDouble("resetThresh", 1.0);
        penaltyMult = params.getDefaultDouble("penaltyMult", 1.0);
        penaltyPow = params.getDefaultDouble("penaltyPow", 1.0);
        
    }

    @Override
    public void filter(Value in) {
        
        // Grab current time
        long currTimeMs = System.currentTimeMillis();
        
        
        // Calculate normalized value
        double inVal = ((double) in.getValue()) / ((double) Integer.MAX_VALUE);
        
        
        // Update counter according to inVal
        if (inVal > resetThresh) {
//            System.out.println("    Holdoff counter reset - was " + counter);
            counter = 0.0;
        }
        else if (inVal > penaltyThresh){
            double range = resetThresh - penaltyThresh;
            counter = counter - penaltyMult * (currTimeMs - prevTimeMs) * Math.pow( (inVal - penaltyThresh) / range, penaltyPow );
            counter = Math.max(counter, 0.0);
        }
        else
            counter += (currTimeMs - prevTimeMs);
        
        
        // Update maxHoldoffCounter
        maxHoldoffCounter += (currTimeMs - prevTimeMs);
        
        
        // Set output flag indicating if hold-off period has passed
        if (counter > currHoldoffLenMs || (maxHoldoffCounter > maxHoldoffLenMs))
            in.setValue(Integer.MAX_VALUE);
        else
            in.setValue((int)counter);
        
//        System.out.println("    Counter = " + counter);
        
        // Update prevTime for next call
        prevTimeMs = currTimeMs;
    }


    public void startHoldoff() {
        // Start a hold-off with the default length
        startHoldoff( 0.0 );
    }


    public void startHoldoff(double holdoffLenAdjustMs) {
        
        prevTimeMs = System.currentTimeMillis();
        counter = 0.0;
        maxHoldoffCounter = 0.0;
        
        // Possibly adjust the length of the hold-off period, relative to the default
        currHoldoffLenMs = holdoffLenMs + holdoffLenAdjustMs;
        
//        System.out.println("  Holdoff counter started - hold-off time: " + currHoldoffLenMs);
    }

}
