package net.electroland.eio.vchannels;

import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.eio.VirtualChannel;
import net.electroland.utils.ParameterMap;

public class DifferenceOfInputsVirtualChannel extends VirtualChannel {

    @Override
    public void configure(ParameterMap params) {
    }

    @Override
    public Value read(ValueSet inputValues) {
        // Is there a better way to do this? Will this even work?
        Value[] inputVals = (Value[]) inputValues.values().toArray();
        Value output = new Value( inputVals[0].getValue() );
        output.setValue( output.getValue() - inputVals[1].getValue() );
        return output;
    }
}