package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * 
 * 
 * The data supplied to this filter should be approximately zero-mean.
 * 
 * @author Sean
 *
 */

public class PersonPresenceFilter implements Filter {

    private StaOverLtaFilter staOverLta;
    private double maxStaOverLtaOutputVal;
    
    private HalfBiquad hpFilt;

    @Override
    public void configure(ParameterMap params) {
        
        // Parameters are hard-coded. These have been tuned for this particular 
        //    application through external analysis and investigation using MATLAB,
        //    and should not be changed.
        
        maxStaOverLtaOutputVal = 150.0;
        
        ParameterMap staOverLtaParams = new ParameterMap();
        staOverLtaParams.put("sta_tauInSamples","0.5");
        staOverLtaParams.put("lta_tauInSamples","37.5");
        staOverLtaParams.put("ltaDelay","12");
        staOverLtaParams.put("ltaMinVal","0.01");
        staOverLtaParams.put("ltaMaxVal","0.3");
        staOverLtaParams.put("maxOutputVal", Double.toString(maxStaOverLtaOutputVal));
        staOverLta = new StaOverLtaFilter();
        staOverLta.configure( staOverLtaParams );
        
        ParameterMap hpParams = new ParameterMap();
        hpParams.put("b0", "1.158384440324536246436082365107722580432891845703125 ");
        hpParams.put("b1", "-1.158384440324536246436082365107722580432891845703125 ");
        hpParams.put("a1", "-0.15838444032453635745838482762337662279605865478515625");
        hpFilt = new HalfBiquad();
        hpFilt.configure( hpParams );
        
    }

    @Override
    public void filter(Value in) {
        
        staOverLta.filter( in );
        
        hpFilt.filter( in );
        
        in.setValue( Math.max( in.getValue(), 0 ) );
        
        // Because we have no more filtering / processing to do, we can now clip the output value 
        //    without concern. We want to use thresholds that correspond to sta / lta values in the
        //    range [0.0 1.0] with the current set up, so we will map that range to [0 Integer.MAX_VALUE]
        //    to retain maximum precision.
        double outValScaled = maxStaOverLtaOutputVal * in.getValue();
        if (outValScaled > Integer.MAX_VALUE)
            outValScaled = Integer.MAX_VALUE;
        in.setValue( (int) outValScaled );
    }

}
