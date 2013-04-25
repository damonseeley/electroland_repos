package net.electroland.eio;

import java.util.ArrayList;
import java.util.Collection;

public abstract class VirtualChannel extends InputChannel {

    protected Collection<InputChannel> inputChannels;

    abstract public Value read(ValueSet inputValues);

    public void addChannel(InputChannel ic){
        if (inputChannels == null){
            inputChannels = new ArrayList<InputChannel>();
        }
        inputChannels.add(ic);
    }
}