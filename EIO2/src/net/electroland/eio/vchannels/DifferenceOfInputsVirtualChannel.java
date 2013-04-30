package net.electroland.eio.vchannels;

import net.electroland.eio.InputChannel;
import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.eio.VirtualChannel;
import net.electroland.utils.ParameterMap;

public class DifferenceOfInputsVirtualChannel extends VirtualChannel {

    @Override
    public void configure(ParameterMap params) {
    }

    /**
     * Takes exactly two inputs, returns the difference. Assumes inputs are
     * ordered as they are in the io.properties spec under vchannel.$ichannels
     * properties. Always returns ichannel[0] - ichannel[1].
     * 
     * @throws RuntimeException if the number of inputValues != 2;
     */
    @Override
    public Value processInputs(ValueSet inputValues) {

        if (this.getChannels().size() != 2){
            throw new RuntimeException("DifferenceOfInputsVirtualChannel requires exactly 2 inputValues.");

        }else{

            InputChannel one = this.getChannels().get(0);
            InputChannel two = this.getChannels().get(1);

            Value output = new Value( inputValues.get(one).getValue() );
            output.setValue( output.getValue() - inputValues.get(two).getValue() );
            return output;
        }
    }
}