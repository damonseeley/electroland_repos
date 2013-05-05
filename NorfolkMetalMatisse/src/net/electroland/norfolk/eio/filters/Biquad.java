package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class Biquad implements Filter {

    private double b0, b1, b2;
    private double a1, a2;          // Assume a0 == 1
    private double z1, z2;
    
    @Override
    public void configure(ParameterMap params) {
        b0 = params.getRequiredDouble("b0");
        b1 = params.getRequiredDouble("b1");
        b2 = params.getRequiredDouble("b2");
        
        a1 = params.getRequiredDouble("a1");
        a2 = params.getRequiredDouble("a2");
        
        z1 = 0.0;
        z2 = 0.0;
    }

    @Override
    public void filter(Value in) {
        double input = in.getValue();
        
        double output = z1 + b0 * input;
        z1 = z2 + b1 * input - a1 * output;
        z2 = b2 * input - a2 * output;
        
    	in.setValue((int)output);
    }

}
