package net.electroland.eio;

import java.util.ArrayList;
import java.util.Collection;

import net.electroland.utils.ParameterMap;

public abstract class VirtualChannel extends InputChannel {

    protected Collection<InputChannel> inputChannels;

    abstract public void configure(ParameterMap params);
    
    abstract public Value read(ValueSet inputValues);

    public void addChannel(InputChannel ic){
        if (inputChannels == null){
            inputChannels = new ArrayList<InputChannel>();
        }
        inputChannels.add(ic);
    }
}