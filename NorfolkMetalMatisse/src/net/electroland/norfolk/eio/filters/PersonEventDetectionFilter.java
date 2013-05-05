package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * This class implements a filter that is to be applied to the mean of the L and
 * R channel signals to generate a detector signal for use in detecting PersonEvents
 * from sensor data. This filter runs two STA / LTA filters with different parameters
 * in parallel and takes the maximum of the two outputs as its output in order to
 * achieve both fast response times and improved response to events when the long
 * term average is high (there has recently been activity in the sensor signals).
 * 
 * The data supplied to this filter should be approximately zero-mean.
 * 
 * @author Sean
 *
 */

public class PersonEventDetectionFilter implements Filter {

    private StaOverLtaFilter staOverLta1;
    private StaOverLtaFilter staOverLta2;
    private double maxStaOverLtaOutputVal;
    
    private HalfBiquad hpFilt1;
    private HalfBiquad hpFilt2;

    @Override
    public void configure(ParameterMap params) {
        
        // Parameters are hard-coded. These have been tuned for this particular 
        //    application through external analysis and investigation using MATLAB,
        //    and should not be changed.
        
        maxStaOverLtaOutputVal = 150.0;
        
        ParameterMap params1 = new ParameterMap();
        params1.put("sta_tauInSamples", "0.5");
        params1.put("lta_tauInSamples", "50.0");
        params1.put("ltaDelay", "10");
        params1.put("ltaMinVal", "0.01");
        params1.put("ltaMaxVal", "0.3");
        params1.put("maxOutputVal", Double.toString(maxStaOverLtaOutputVal));
        staOverLta1 = new StaOverLtaFilter();
        staOverLta1.configure( params1 );
        
        ParameterMap params2 = new ParameterMap();
        params2.put("sta_tauInSamples", "0.5");
        params2.put("lta_tauInSamples", "25.0");
        params2.put("ltaDelay", "15");
        params2.put("ltaMinVal", "0.01");
        params2.put("ltaMaxVal", "0.3");
        params2.put("maxOutputVal", Double.toString(maxStaOverLtaOutputVal));
        staOverLta2 = new StaOverLtaFilter();
        staOverLta2.configure( params2 );
        
        ParameterMap hpParams1 = new ParameterMap();
        hpParams1.put("b0", "0.926707552259629085966707862098701298236846923828125");
        hpParams1.put("b1", "-0.926707552259629085966707862098701298236846923828125");
        hpParams1.put("a1", "-0.15838444032453635745838482762337662279605865478515625");
        hpFilt1 = new HalfBiquad();
        hpFilt1.configure(hpParams1);
        
        ParameterMap hpParams2 = new ParameterMap();
        hpParams2.put("b0", "0.77225629354969083095738824340514838695526123046875");
        hpParams2.put("b1", "-0.77225629354969083095738824340514838695526123046875");
        hpParams2.put("a1", "-0.15838444032453635745838482762337662279605865478515625");
        hpFilt2 = new HalfBiquad();
        hpFilt2.configure(hpParams2);
    }

    @Override
    public void filter(Value in) {
        
        Value filt1Val = new Value( in.getValue() );
        Value filt2Val = new Value( in.getValue() );
        
        staOverLta1.filter( filt1Val );
        staOverLta2.filter( filt2Val );
        
        hpFilt1.filter( filt1Val );
        hpFilt2.filter( filt2Val );
        
        in.setValue( Math.max( Math.max( filt1Val.getValue(), filt2Val.getValue() ), 0 ) );
        
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
