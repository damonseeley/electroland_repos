package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class EventDetectionConditioningFilter implements Filter {

    private HalfBiquad halfBq = new HalfBiquad();
    private Biquad bq = new Biquad();
    
    @Override
    public void configure(ParameterMap map) {
        
        ParameterMap halfBqParams = new ParameterMap();
        halfBqParams.put("b0", "4.054345541135877084570893202908337116241455078125");
        halfBqParams.put("b1", "-4.054345541135877084570893202908337116241455078125");
        halfBqParams.put("a1", "-0.15838444032453635745838482762337662279605865478515625");
        halfBq.configure(halfBqParams);
        
        ParameterMap bqParams = new ParameterMap();
        bqParams.put("b0","0.7547627247472146194695596932433545589447021484375");
        bqParams.put("b1","0.93293803467051983346891574910841882228851318359375");
        bqParams.put("b2","0.7547627247472146194695596932433545589447021484375");
        bqParams.put("a1","0.93293803467051983346891574910841882228851318359375");
        bqParams.put("a2","0.5095254494944290168945144614554010331630706787109375");
        bq.configure(bqParams);
        
    }

    @Override
    public void filter(Value in) {
        halfBq.filter(in);
        bq.filter(in);
    }

}
