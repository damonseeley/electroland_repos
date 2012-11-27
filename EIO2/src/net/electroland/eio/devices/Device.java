package net.electroland.eio.devices;

import java.util.Map;

import net.electroland.eio.Value;
import net.electroland.utils.ParameterMap;

public abstract class Device {

    public Device(ParameterMap params){}

    abstract public Map<Channel, Value> read();

    abstract public void write(Map<Channel, Value> values);
}
