package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

/**
 * This class implements a basic circular delay line. 
 * 
 * Currently, the amount of delay is static during the life of the object.
 * 
 * @author Sean
 *
 */

public class DelayLine implements Filter {

    private int[] buffer;
    private int writeInd = 0;
    private int readInd = 0;
    
    @Override
    public void configure(ParameterMap params) {
        
        // Make the delay line one sample longer to enable flexible read/write order
        //    (although it's not currently used...)
        int delay = params.getRequiredInt("delay") + 1;
        
        buffer = new int[delay]; // Should initialize contents to 0 by default
        
        writeInd = 0;
        readInd = 1;
    }

    @Override
    public void filter(Value in) {
        
        buffer[writeInd] = in.getValue();
        in.setValue(buffer[readInd]);
        
        writeInd = (writeInd+1) % buffer.length;
        readInd = (readInd+1) % buffer.length;
        
    }

    public void setAllValues(int inVal) {
        
        for (int i=0; i < buffer.length; i++)
            buffer[i] = inVal;
    }

}
