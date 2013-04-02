package net.electroland.eio;

import net.electroland.utils.ParameterMap;

public abstract class Device {

    public Device(ParameterMap params){}

    abstract public ValueSet read();

    abstract public void write(ValueSet values);

    abstract public void close();

    abstract public InputChannel patch(ParameterMap channelParams);
}