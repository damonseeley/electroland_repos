package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class HalfBiquad implements Filter {

    private double b0, b1;
    private double a1;          // Assume a0 == 1
    private double z1;
    
    @Override
    public void configure(ParameterMap params) {
        b0 = params.getRequiredDouble("b0");
        b1 = params.getRequiredDouble("b1");
        a1 = params.getRequiredDouble("a1");
        z1 = 0.0;
    }

    @Override
    public void filter(Value in) {
        double input = in.getValue();
        
        double output = z1 + b0 * input;
        z1 = b1 * input - a1 * output;
        
//        if (output > Integer.MAX_VALUE)
//            throw new RuntimeException("Overflow occurred in HalfBiquad.");
        
        // Clip output value to avoid overflow
        if (output > Integer.MAX_VALUE)
            output = Integer.MAX_VALUE;
        else if (output < Integer.MIN_VALUE)
            output = Integer.MIN_VALUE;
        
    	in.setValue((int)output);
    }

}
