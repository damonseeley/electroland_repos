package net.electroland.eio.vchannels;

import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.eio.VirtualChannel;
import net.electroland.utils.ParameterMap;

public class SampleVirtualChannel extends VirtualChannel {

    @Override
    public void configure(ParameterMap params) {
        // nothing require for this one.
    }

    /**
     * Just returns the average of all connected channels.
     */
    @Override
    public Value read(ValueSet inputValues) {

        int total = 0;

        for (Value v : inputValues.values()){
            total += v.getValue();
        }

        return new Value((int)(total / (float)inputValues.values().size()));
    }
}