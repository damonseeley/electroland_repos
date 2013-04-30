package net.electroland.eio;

import java.util.ArrayList;
import java.util.List;

import net.electroland.utils.ParameterMap;

public abstract class VirtualChannel extends InputChannel {

    protected List<InputChannel> inputChannels;

    abstract public void configure(ParameterMap params);

    abstract public Value processInputs(ValueSet inputValues);

    public void addChannel(InputChannel ic){
        if (inputChannels == null){
            inputChannels = new ArrayList<InputChannel>();
        }
        inputChannels.add(ic);
    }

    public List<InputChannel> getChannels(){
        return inputChannels;
    }
}