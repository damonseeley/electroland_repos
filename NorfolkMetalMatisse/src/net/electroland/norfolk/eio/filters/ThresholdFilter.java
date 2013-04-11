package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * Searches for values greater than a threshold. If you hit it, that's a
 * positive (1) read. If not, it's a negative (0). After you generate a positive
 * read, this filter goes dormant for millisTimeoutAfterPositive millisenconds.
 * 
 * Not sure if we'll needed it, but trueIfAboveThreshold can be used to switch
 * from searching for values that exceed the threshold to values that are below
 * it (incase we get into inversion scenarios).
 * 
 * @author bradley
 *
 */
public class ThresholdFilter implements Filter {

    int  threshold;
    int  aboveInclusiveValue, belowExclusiveValue;

    @Override
    public void configure(ParameterMap map) {
        threshold            = map.getRequiredInt("threshold");
        aboveInclusiveValue  = map.getRequiredInt("aboveInclusiveValue");
        belowExclusiveValue           = map.getRequiredInt("belowExclusiveValue");
    }

    @Override
    public void filter(Value in) {
        in.setValue(in.getFilteredValue()  >= threshold ? aboveInclusiveValue : belowExclusiveValue);
    }
}