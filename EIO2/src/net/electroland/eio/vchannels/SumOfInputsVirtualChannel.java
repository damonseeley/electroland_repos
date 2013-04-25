package net.electroland.eio.vchannels;

import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.eio.VirtualChannel;
import net.electroland.utils.ParameterMap;

public class SumOfInputsVirtualChannel extends VirtualChannel {

    @Override
    public void configure(ParameterMap params) {
    }

    @Override
    public Value read(ValueSet inputValues) {
        Value output = new Value(0);
        for(Value currVal : inputValues.values())
            output.setValue(output.getValue() + currVal.getValue());
        return output;
    }
}