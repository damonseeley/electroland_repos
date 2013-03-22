package net.electroland.eio;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public abstract class Device {

    public Device(ParameterMap params){}

    abstract public Map<InputChannel, Value> read();

    abstract public void write(Map<OutputChannel, Value> values);

    abstract public void close();

    abstract public InputChannel addInputChannel(ParameterMap channelParams);
}