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
    public Value processInputs(ValueSet inputValues) {
        if (inputValues.values().size() != 2){
            throw new RuntimeException("DifferenceOfInputsVirtualChannel requires exactly 2 inputValues.");
        }else{
            Value[] inputVals = (Value[]) inputValues.values().toArray();
            Value output = new Value( inputVals[0].getValue() );
            output.setValue( output.getValue() - inputVals[1].getValue() );
            return output;
        }
    }
}