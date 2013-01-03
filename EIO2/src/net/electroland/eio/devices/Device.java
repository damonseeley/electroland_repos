package net.electroland.eio.devices;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public abstract class Device {

    public Device(ParameterMap params){}

    abstract public Map<Channel, Object> read();

    abstract public void write(Map<Channel, Object> values);
}
