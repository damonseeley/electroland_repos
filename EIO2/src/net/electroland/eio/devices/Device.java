package net.electroland.eio.devices;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public abstract class Device {

    public Device(ParameterMap params){}

    abstract public Map<InputChannel, Object> read();

    abstract public void write(Map<OutputChannel, Object> values);

    abstract public InputChannel addInputChannel(ParameterMap channelParams);
}